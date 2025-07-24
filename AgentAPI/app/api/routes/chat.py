from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from typing import Dict, Any, Optional
from langchain_core.messages import HumanMessage, SystemMessage, AIMessageChunk
from dataclasses import dataclass, field
import json
import time
import uuid
import hashlib
import logging

from agents.base_agents.base_agent import BaseAgent
from agents.base_agents.base_state import BaseState
from langchain_core.runnables import RunnableConfig
from agents import get_agent_by_id, AGENT_CLASS_MAP
from tools.agents_as_tools.knowledge_search.knowledge_search import knowledge_search_agent  
from tools.agents_as_tools.web_search.web_search import web_search_scrape_agent
from tools.agents_as_tools.mcp.mcp_search import mcp_agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

@dataclass
class ReasoningCollector:
    reasoning_parts: list = field(default_factory=list)
    
    def add_reasoning(self, content: str):
        self.reasoning_parts.append(content)
    
    def get_reasoning_content(self) -> str:
        return "\n".join(self.reasoning_parts)

TOOL_AGENT_MAP = {
    "knowledge_search": knowledge_search_agent,
    "web_search_scrape_agent": web_search_scrape_agent,
    "mcp_agent": mcp_agent,
}

class ChatCompletionService:
    def __init__(self):
        self.completion_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
        
    def validate_request(self, request: dict) -> str:
        model = request.get("model")
        if not model:
            raise HTTPException(status_code=400, detail="No model specified")
        
        if model not in AGENT_CLASS_MAP:
            raise HTTPException(status_code=400, detail=f"Model Not Found : {model}")
            
        messages = request.get("messages")
        if not messages:
            raise HTTPException(status_code=400, detail="No messages provided")
        
        extra_body = request
        if not extra_body:
            raise HTTPException(status_code=400, detail="No extra_body provided")
        
        org_id = extra_body.get("org_id")
        if not org_id:
            raise HTTPException(status_code=400, detail="No org_id provided in extra_body")
            
        user_id = extra_body.get("user_id")
        if not user_id:
            raise HTTPException(status_code=400, detail="No user_id provided in extra_body")
            
        thread_id = extra_body.get("thread_id")
        if not thread_id:
            raise HTTPException(status_code=400, detail="No thread_id provided in extra_body")
        
        # checkpoint_id is optional but we can validate if provided
        # checkpoint_id = extra_body.get("checkpoint_id")
        
        return model

    def setup_agent_with_tools(self, model_id: str, config: RunnableConfig = None, tools: Optional[list] = []) -> BaseAgent:
        agent_cls = get_agent_by_id(model_id)
        if not agent_cls:
            raise HTTPException(status_code=400, detail=f"Unknown agent: {model_id}")

        agent_tools = []
        if tools:
            for tool_obj in tools:
                if isinstance(tool_obj, dict):
                    if tool_obj.get("type") == "function" and "function" in tool_obj:
                        tool_name = tool_obj["function"].get("name")
                        agent = TOOL_AGENT_MAP.get(tool_name)
                        if agent:
                            agent_tools.append(agent)
                elif isinstance(tool_obj, str):
                    agent = TOOL_AGENT_MAP.get(tool_obj)
                    if agent:
                        agent_tools.append(agent)
        
        agent_instance = agent_cls(config=config)
        if agent_tools:
            agent_instance = agent_instance.bind_tools(agent_tools)
        return agent_instance

    def build_input_data_and_config(self, request: dict) -> tuple:
        extra_body = request
        
        org_id = f"{extra_body.get('org_id')}${hashlib.sha256(str(request.get('model')).encode()).hexdigest()}" if extra_body.get('org_id') else None
        
        input_data = {
            "messages": [
                HumanMessage(content=m["content"]) if m["role"] == "user" 
                else SystemMessage(content=m["content"])
                for m in request.get("messages", [])
            ],
            "user_id": extra_body.get("user_id"),
            "org_id": org_id
        }
        
        config = {"configurable": {}}
        thread_id = extra_body.get("thread_id")
        user_id = extra_body.get("user_id")
        checkpoint_id = extra_body.get("checkpoint_id")
        
        if thread_id:
            config["configurable"]["thread_id"] = thread_id
        if user_id:
            config["configurable"]["user_id"] = user_id
        if org_id:
            config["configurable"]["org_id"] = org_id
        if checkpoint_id:
            config["configurable"]["checkpoint_id"] = checkpoint_id
        
        return input_data, config

    async def create_streaming_response(self, agent:BaseAgent, input_data: dict, config: dict, model_id: str):
        async def event_stream():
            try:
                collector = ReasoningCollector()
                                
                async for stream_mode, chunk in agent.astream(input_data, config=config, stream_mode=["messages", "custom"]):
                    
                    if stream_mode == "messages":
                        message_chunk, metadata = chunk
                        if hasattr(message_chunk, "content") and message_chunk.content:
                            node_name = metadata.get("langgraph_node", "unknown")
                            
                            if len(str(node_name).split('$')) == 2:
                                
                                for char in message_chunk.content:
                                    if char == "":
                                        continue
                                    response_chunk = {
                                        "id": self.completion_id,
                                        "object": "chat.completion.chunk",
                                        "created": int(time.time()),
                                        "model": model_id,
                                        "choices": [{
                                            "index": 0,
                                            "delta": {"content": char},
                                            "finish_reason": None
                                        }],
                                        "usage": None
                                    }
                                    yield f"data: {json.dumps(response_chunk, ensure_ascii=False)}\n\n"
                            else:
                                reasoning_text = message_chunk.content
                                if node_name == "tools":
                                    continue
                                for char in reasoning_text:
                                    response_chunk = {
                                        "id": self.completion_id,
                                        "object": "chat.completion.chunk",
                                        "created": int(time.time()),
                                        "model": model_id,
                                        "choices": [{
                                            "index": 0,
                                            "delta": {"reasoning_content": char},
                                            "logprobs": None,
                                            "finish_reason": None
                                        }],
                                        "usage": None
                                    }
                                    yield f"data: {json.dumps(response_chunk, ensure_ascii=False)}\n\n"
                                
                                collector.add_reasoning(reasoning_text)
                    
                    elif stream_mode == "custom":
                        if isinstance(chunk, dict):
                            reasoning_text = str("executing tool")
                        else:
                            reasoning_text = str(chunk)
                        
                        for char in reasoning_text:
                            response_chunk = {
                                "id": self.completion_id,
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": model_id,
                                "choices": [{
                                    "index": 0,
                                    "delta": {"reasoning_content": char},
                                    "logprobs": None,
                                    "finish_reason": None
                                }],
                                "usage": None
                            }
                            yield f"data: {json.dumps(response_chunk, ensure_ascii=False)}\n\n"
                        
                        collector.add_reasoning(reasoning_text)
                
                yield "data: [DONE]\n\n"
                
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                error_chunk = {
                    "id": self.completion_id,
                    "object": "error",
                    "created": int(time.time()),
                    "error": {"message": "Internal processing error", "type": "agent_error"}
                }
                yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
        
        return StreamingResponse(event_stream(), media_type="text/event-stream")

    async def create_non_streaming_response(self, agent:BaseAgent, input_data: dict, config: dict, model_id: str):
        try:
            collector = ReasoningCollector()
            final_content = ""
            
            async for stream_mode, chunk in agent.astream(input_data, config=config, stream_mode=["updates", "custom"]):
                
                if stream_mode == "updates":
                    if isinstance(chunk, dict):
                        for node_name, node_data in chunk.items():
                            if isinstance(node_data, dict) and "messages" in node_data:
                                messages = node_data["messages"]
                                if messages and len(messages) > 0:
                                    last_message = messages[-1]
                                    if hasattr(last_message, "content") and last_message.content:
                                        if node_name == "__end__":
                                            final_content = last_message.content
                                        collector.add_reasoning(last_message.content)
                
                elif stream_mode == "custom":
                    if isinstance(chunk, dict):
                        custom_data = json.dumps(chunk, ensure_ascii=False)
                        collector.add_reasoning(custom_data)
                    else:
                        collector.add_reasoning(str(chunk))
            
            return {
                "id": self.completion_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_id,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "reasoning_content": collector.get_reasoning_content(),
                        "content": final_content or "No response generated",
                        "tool_calls": []
                    },
                    "logprobs": None,
                    "finish_reason": "stop",
                    "stop_reason": None
                }],
                "usage": {
                    "prompt_tokens": 0,
                    "total_tokens": 0,
                    "completion_tokens": 0,
                    "prompt_tokens_details": None
                },
                "prompt_logprobs": None,
                "kv_transfer_params": None
            }
        except Exception as e:
            logger.error(f"Non-streaming error: {e}")
            raise HTTPException(status_code=500, detail="Internal processing error")

@router.post("/v1/chat/completions")
async def agent_completions(request: dict):
    logger.info(f"Received chat completion request for model: {request.get('model')}")
    
    try:
        service = ChatCompletionService()
        model_id = service.validate_request(request)
        input_data, config = service.build_input_data_and_config(request)
        
        agent_instance = service.setup_agent_with_tools(model_id, config, request.get("tools"))
        agent = await agent_instance.compile(name=model_id)
        
        stream = request.get("stream", False)
        if stream:
            return await service.create_streaming_response(agent, input_data, config, model_id)
        else:
            return await service.create_non_streaming_response(agent, input_data, config, model_id)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in agent_completions: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")