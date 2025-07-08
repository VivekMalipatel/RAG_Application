from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from app.agents import AVAILABLE_AGENTS

router = APIRouter()

@router.get("/v1/agents/list")
async def list_agents():
    """List all available agents with id, name, and description."""
    return {"agents": AVAILABLE_AGENTS}

class Message(BaseModel):
    role: str
    content: str

class AgentCompletionRequest(BaseModel):
    agent: str = Field(..., description="Agent name or model identifier.")
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1024
    stream: Optional[bool] = False
    user: Optional[str] = None
    tools: Optional[List[str]] = None
    # Add more OpenAI-compatible fields as needed

class AgentCompletionResponse(BaseModel):
    id: str
    object: str = "agent.completion"
    created: int
    agent: str
    choices: List[Dict[str, Any]]
    usage: Optional[Dict[str, int]] = None

from app.agents import get_agent_by_id
from langchain_core.messages import HumanMessage, SystemMessage
from fastapi.responses import StreamingResponse
import json
import time
import uuid

@router.post("/v1/agents/completions", response_model=AgentCompletionResponse)
async def agent_completions(request: AgentCompletionRequest):
    agent_id = request.agent
    agent_cls = get_agent_by_id(agent_id)
    if not agent_cls:
        raise HTTPException(status_code=400, detail=f"Unknown agent: {agent_id}")
    agent = agent_cls().compile()
    state = {"messages": [
        HumanMessage(content=m.content) if m.role == "user" else SystemMessage(content=m.content)
        for m in request.messages
    ]}

    if request.stream:
        async def event_stream():
            try:
                idx = 0
                async for event in agent.astream_events(state, version="v2"):
                    # Stream LLM token events (similar to OpenAI streaming)
                    if (event.get("event") == "on_chat_model_stream" and 
                        event.get("data", {}).get("chunk")):
                        chunk = event["data"]["chunk"]
                        content = getattr(chunk, "content", "")
                        if content:  # Only send non-empty content
                            data = {
                                "id": f"agentcmpl-{uuid.uuid4().hex[:16]}",
                                "object": "agent.completion.chunk",
                                "created": int(time.time()),
                                "agent": agent_id,
                                "choices": [{
                                    "index": 0,
                                    "delta": {"content": content},
                                    "finish_reason": None
                                }]
                            }
                            yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                            idx += 1
                    # Stream final result
                    elif (event.get("event") == "on_chain_end" and 
                          event.get("name") == "LangGraph"):
                        data = {
                            "id": f"agentcmpl-{uuid.uuid4().hex[:16]}",
                            "object": "agent.completion.chunk",
                            "created": int(time.time()),
                            "agent": agent_id,
                            "choices": [{
                                "index": 0,
                                "delta": {},
                                "finish_reason": "stop"
                            }]
                        }
                        yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                error_data = {
                    "id": f"agentcmpl-{uuid.uuid4().hex[:16]}",
                    "object": "error",
                    "created": int(time.time()),
                    "error": {"message": str(e), "type": "agent_error"}
                }
                yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # Non-streaming case
    try:
        result = await agent.ainvoke(state)
        output = result["messages"][-1].content if result.get("messages") else ""
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")
    return AgentCompletionResponse(
        id=f"agentcmpl-{uuid.uuid4().hex[:16]}",
        created=int(time.time()),
        agent=agent_id,
        choices=[{
            "index": 0,
            "message": {"role": "assistant", "content": output},
            "finish_reason": "stop"
        }],
        usage=None
    )
