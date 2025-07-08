from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from .models import Message

router = APIRouter()

class AgentCompletionRequest(BaseModel):
    agent: str = Field(..., description="Agent name or model identifier.")
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1024
    stream: Optional[bool] = False
    user: Optional[str] = None
    tools: Optional[List[str]] = None
    # Add more OpenAI-compatible fields as needed

from app.agents import get_agent_by_id
from langchain_core.messages import HumanMessage, SystemMessage
from fastapi.responses import StreamingResponse
import json
import time
import uuid

def openai_stream_event_handler(agent, state, model_id):
    async def event_stream():
        try:
            accumulated_content = ""
            completion_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
            async for event in agent.astream_events(state, version="v2"):
                event_type = event.get("event")
                event_name = event.get("name", "")
                event_data = event.get("data", {})
                metadata = event.get("metadata", {})
                if event_type == "on_chat_model_stream":
                    chunk_data = event_data.get("chunk")
                    if chunk_data and hasattr(chunk_data, "content"):
                        content = chunk_data.content
                        if content:
                            accumulated_content += content
                            response_chunk = {
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": model_id,
                                "choices": [{
                                    "index": 0,
                                    "delta": {"content": content},
                                    "finish_reason": None
                                }],
                                "usage": None
                            }
                            yield f"data: {json.dumps(response_chunk, ensure_ascii=False)}\n\n"
                elif event_type == "on_chain_start":
                    node_name = metadata.get("langgraph_node", "unknown")
                    print(f"Node started: {node_name}")
                elif event_type == "on_chain_end":
                    node_name = metadata.get("langgraph_node", "unknown")
                    print(f"Node completed: {node_name}")
                    if event_name == "LangGraph":
                        final_chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": model_id,
                            "choices": [{
                                "index": 0,
                                "delta": {},
                                "finish_reason": "stop"
                            }],
                            "usage": {
                                "prompt_tokens": 0,
                                "completion_tokens": len(accumulated_content.split()),
                                "total_tokens": len(accumulated_content.split())
                            }
                        }
                        yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            print(f"Streaming error: {e}")
            error_chunk = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:16]}",
                "object": "error",
                "created": int(time.time()),
                "error": {"message": str(e), "type": "agent_error"}
            }
            yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
    return event_stream

def openai_nonstream_completion(agent, state, model_id):
    async def nonstream():
        try:
            print(f"About to call agent.ainvoke with state: {state}")
            result = await agent.ainvoke(state)
            print(f"ainvoke result: {result}")
            if result and "messages" in result and result["messages"]:
                last_msg = result["messages"][-1]
                if hasattr(last_msg, 'content'):
                    output = last_msg.content
                else:
                    output = str(last_msg)
            else:
                output = "No response generated"
        except Exception as invoke_error:
            print(f"Error during agent.ainvoke: {invoke_error}")
            output = f"Agent processing error: {str(invoke_error)}"
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:16]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_id,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": output},
                "finish_reason": "stop"
            }],
            "usage": None
        }
    return nonstream

@router.post("/v1/chat/completions")
async def agent_completions(request: dict):
    try:
        print(f"Received request: {request}")  # Debug log
        model_id = request.get("model") or request.get("agent")
        if not model_id:
            raise HTTPException(status_code=400, detail="No model specified")
        agent_cls = get_agent_by_id(model_id)
        if not agent_cls:
            raise HTTPException(status_code=400, detail=f"Unknown agent/model: {model_id}")
        agent = agent_cls().compile()
        messages = request.get("messages", [])
        if not messages:
            raise HTTPException(status_code=400, detail="No messages provided")
        state = {"messages": [
            HumanMessage(content=m["content"]) if m["role"] == "user" else SystemMessage(content=m["content"])
            for m in messages
        ]}
        stream = request.get("stream", False)
        if stream:
            return StreamingResponse(openai_stream_event_handler(agent, state, model_id)(), media_type="text/event-stream")
        return await openai_nonstream_completion(agent, state, model_id)()
    except Exception as e:
        print(f"Error in agent_completions: {str(e)}")  # Debug log
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

