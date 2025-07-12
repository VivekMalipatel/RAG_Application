from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from models import Message
from agents import get_agent_by_id
from langchain_core.messages import HumanMessage, SystemMessage
from fastapi.responses import StreamingResponse
import json
import time
import uuid
from dataclasses import dataclass, field

router = APIRouter()

@dataclass
class EventTracker:
    """Tracks various event states and metrics"""
    node_states: Dict[str, str] = field(default_factory=dict)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    streaming_content: List[str] = field(default_factory=list)
    current_node: Optional[str] = None
    total_tokens: int = 0
    start_time: float = field(default_factory=time.time)
    errors: List[str] = field(default_factory=list)

def comprehensive_openai_stream_event_handler(agent, state, model_id, enable_progress_updates=True):

    async def event_stream():
        try:
            accumulated_content = ""
            completion_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
            tracker = EventTracker()
            async for event in agent.astream_events(state, version="v2"):
                event_type = event.get("event")

                if event_type == "on_chat_model_start":
                    await _handle_chat_model_start(event, completion_id, model_id, tracker)
                elif event_type == "on_chat_model_stream":
                    chunk_content = await _handle_chat_model_stream(
                        event, completion_id, model_id, tracker, accumulated_content
                    )
                    if chunk_content:
                        accumulated_content += chunk_content
                        response_chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": model_id,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": chunk_content},
                                "finish_reason": None
                            }],
                            "usage": None
                        }
                        yield f"data: {json.dumps(response_chunk, ensure_ascii=False)}\n\n"
                elif event_type == "on_chat_model_end":
                    await _handle_chat_model_end(event, completion_id, model_id, tracker)

                elif event_type == "on_chain_start":
                    await _handle_chain_start(event, completion_id, model_id, tracker, enable_progress_updates)
                elif event_type == "on_chain_end":
                    completion_signal = await _handle_chain_end(
                        event, completion_id, model_id, tracker, accumulated_content
                    )
                    if completion_signal:
                        yield f"data: {json.dumps(completion_signal, ensure_ascii=False)}\n\n"
                elif event_type == "on_chain_error":
                    await _handle_chain_error(event, completion_id, model_id, tracker)

                elif event_type == "on_tool_start":
                    await _handle_tool_start(event, completion_id, model_id, tracker, enable_progress_updates)
                elif event_type == "on_tool_end":
                    await _handle_tool_end(event, completion_id, model_id, tracker, enable_progress_updates)
                elif event_type == "on_tool_error":
                    await _handle_tool_error(event, completion_id, model_id, tracker)

                elif event_type == "on_retriever_start":
                    await _handle_retriever_start(event, completion_id, model_id, tracker)
                elif event_type == "on_retriever_end":
                    await _handle_retriever_end(event, completion_id, model_id, tracker)

                elif event_type == "on_custom_event":
                    await _handle_custom_event(event, completion_id, model_id, tracker)

                elif event_type == "on_prompt_start":
                    await _handle_prompt_start(event, completion_id, model_id, tracker)
                elif event_type == "on_prompt_end":
                    await _handle_prompt_end(event, completion_id, model_id, tracker)

                elif event_type == "on_parser_start":
                    await _handle_parser_start(event, completion_id, model_id, tracker)
                elif event_type == "on_parser_end":
                    await _handle_parser_end(event, completion_id, model_id, tracker)

                elif event_type == "on_retry":
                    await _handle_retry_event(event, completion_id, model_id, tracker)

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

async def _handle_chat_model_start(event, completion_id, model_id, tracker):
    print(f"🤖 LLM Model started: {event.get('name', 'Unknown')}")
    tracker.current_node = "llm_processing"
    tracker.node_states["llm_processing"] = "running"

async def _handle_chat_model_stream(event, completion_id, model_id, tracker, accumulated_content):
    chunk_data = event.get("data", {}).get("chunk")
    if chunk_data and hasattr(chunk_data, "content"):
        content = chunk_data.content
        if content:
            tracker.streaming_content.append(content)
            tracker.total_tokens += len(content.split())
            return content
    return None

async def _handle_chat_model_end(event, completion_id, model_id, tracker):
    print(f"LLM Model completed: {event.get('name', 'Unknown')}")
    tracker.node_states["llm_processing"] = "completed"

async def _handle_chain_start(event, completion_id, model_id, tracker, enable_progress_updates):
    node_name = event.get("metadata", {}).get("langgraph_node", "unknown")
    tracker.current_node = node_name
    tracker.node_states[node_name] = "running"
    print(f"Node started: {node_name}")
    if enable_progress_updates:
        pass

async def _handle_chain_end(event, completion_id, model_id, tracker, accumulated_content):
    node_name = event.get("metadata", {}).get("langgraph_node", "unknown")
    event_name = event.get("name", "")
    if node_name != "unknown":
        tracker.node_states[node_name] = "completed"
        print(f"Node completed: {node_name}")
    if event_name == "LangGraph":
        elapsed_time = time.time() - tracker.start_time
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
                "completion_tokens": tracker.total_tokens,
                "total_tokens": tracker.total_tokens,
                "processing_time_ms": int(elapsed_time * 1000)
            }
        }
        return final_chunk
    return None

async def _handle_chain_error(event, completion_id, model_id, tracker):
    error_info = event.get("data", {}).get("error", {})
    node_name = event.get("metadata", {}).get("langgraph_node", "unknown")
    print(f"Node error in {node_name}: {error_info}")
    tracker.errors.append(f"Node {node_name}: {error_info}")
    tracker.node_states[node_name] = "error"

async def _handle_tool_start(event, completion_id, model_id, tracker, enable_progress_updates):
    tool_name = event.get("name", "unknown_tool")
    tool_input = event.get("data", {}).get("input", {})
    tracker.tool_calls.append({
        "tool": tool_name,
        "input": tool_input,
        "status": "running",
        "node": tracker.current_node,
        "start_time": time.time()
    })
    print(f"Tool started: {tool_name}")
    if enable_progress_updates:
        pass

async def _handle_tool_end(event, completion_id, model_id, tracker, enable_progress_updates):
    tool_name = event.get("name", "unknown_tool")
    tool_output = event.get("data", {}).get("output")
    for tool_call in reversed(tracker.tool_calls):
        if tool_call["tool"] == tool_name and tool_call["status"] == "running":
            tool_call["status"] = "completed"
            tool_call["output"] = tool_output
            tool_call["end_time"] = time.time()
            tool_call["duration"] = tool_call["end_time"] - tool_call["start_time"]
            break
    print(f"Tool completed: {tool_name}")

async def _handle_tool_error(event, completion_id, model_id, tracker):
    tool_name = event.get("name", "unknown_tool")
    error_info = event.get("data", {}).get("error", {})
    for tool_call in reversed(tracker.tool_calls):
        if tool_call["tool"] == tool_name and tool_call["status"] == "running":
            tool_call["status"] = "error"
            tool_call["error"] = error_info
            tool_call["end_time"] = time.time()
            break
    print(f"Tool error: {tool_name} - {error_info}")
    tracker.errors.append(f"Tool {tool_name}: {error_info}")

async def _handle_retriever_start(event, completion_id, model_id, tracker):
    print(f"Retriever started: {event.get('name', 'Unknown')}")

async def _handle_retriever_end(event, completion_id, model_id, tracker):
    print(f"Retriever completed: {event.get('name', 'Unknown')}")

async def _handle_custom_event(event, completion_id, model_id, tracker):
    custom_name = event.get("name", "unknown_custom")
    custom_data = event.get("data", {})
    print(f"Custom event: {custom_name} - {custom_data}")

async def _handle_prompt_start(event, completion_id, model_id, tracker):
    print(f"Prompt processing started: {event.get('name', 'Unknown')}")

async def _handle_prompt_end(event, completion_id, model_id, tracker):
    print(f"Prompt processing completed: {event.get('name', 'Unknown')}")

async def _handle_parser_start(event, completion_id, model_id, tracker):
    print(f"Parser started: {event.get('name', 'Unknown')}")

async def _handle_parser_end(event, completion_id, model_id, tracker):
    print(f"Parser completed: {event.get('name', 'Unknown')}")

async def _handle_retry_event(event, completion_id, model_id, tracker):
    print(f"Retry event: {event.get('name', 'Unknown')}")

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
        enable_progress_updates = request.get("enable_progress_updates", True)
        if stream:
            return StreamingResponse(
                comprehensive_openai_stream_event_handler(
                    agent, state, model_id, enable_progress_updates
                )(),
                media_type="text/event-stream"
            )
        return await openai_nonstream_completion(agent, state, model_id)()
    except Exception as e:
        print(f"Error in agent_completions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

