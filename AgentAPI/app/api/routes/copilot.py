from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
import uuid
import json
import logging
from typing import Any, Dict, List, Optional
from ag_ui.core import (
    RunAgentInput,
    StateSnapshotEvent,
    EventType,
    RunStartedEvent,
    RunFinishedEvent,
    RunErrorEvent,
    TextMessageStartEvent,
    TextMessageEndEvent,
    TextMessageContentEvent,
    ToolCallStartEvent,
    ToolCallEndEvent,
    ToolCallArgsEvent,
    StateDeltaEvent
)
from ag_ui.encoder import EventEncoder
from agents import  AGENT_CLASS_MAP, AVAILABLE_AGENTS
from api.routes.chat import ChatCompletionService
from fastapi import APIRouter
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

from copilotkit import CopilotKitState
from typing import List, Dict, Any, Optional
from pydantic import Field

class StandardAgentState(CopilotKitState):
    """
    Standard state model that can be used across all your agents
    """
    messages: List[Dict[str, Any]] = Field(default_factory=list, description="Conversation history")
    
    tools: List[Dict[str, Any]] = Field(default_factory=list, description="Available tools")
    tool_logs: List[Dict[str, Any]] = Field(default_factory=list, description="Tool execution history")
    
    context: Dict[str, Any] = Field(default_factory=dict, description="Runtime context")
    thread_id: Optional[str] = Field(default=None, description="Conversation thread ID")
    user_id: Optional[str] = Field(default=None, description="User identifier")
    org_id: Optional[str] = Field(default=None, description="Organization identifier")
    
    agent_data: Dict[str, Any] = Field(default_factory=dict, description="Agent-specific state")
    
    be_data: Optional[Any] = Field(default=None, description="Backend API data")
    be_arguments: Optional[Dict[str, Any]] = Field(default=None, description="Backend function arguments")



class StandardAgentWrapper:
    """
    Standard wrapper to make any LangGraph agent compatible with CopilotKit
    """
    
    def __init__(self):
        self.service = ChatCompletionService()
        self.encoder = EventEncoder()
    
    def convert_agui_to_internal(self, agui_input: RunAgentInput, model: str = None) -> dict:
        """Convert AG-UI protocol input to internal request format"""
        # Extract context values
        context_values = {}
        if agui_input.context:
            for ctx in agui_input.context:
                if hasattr(ctx, 'description') and hasattr(ctx, 'value'):
                    if ctx.description == "Organization ID":
                        context_values["org_id"] = ctx.value
                    elif ctx.description == "User ID":
                        context_values["user_id"] = ctx.value
                    elif ctx.description == "Thread ID":
                        context_values["thread_id"] = ctx.value
        
        # Convert tools from objects to strings for internal format
        tools = []
        if agui_input.tools:
            for tool in agui_input.tools:
                if hasattr(tool, 'name'):
                    tools.append(tool.name)
                elif isinstance(tool, dict) and 'name' in tool:
                    tools.append(tool['name'])
                else:
                    tools.append(str(tool))
        
        return {
            "model": model or list(AGENT_CLASS_MAP.keys())[0],
            "messages": [
                {"role": msg.role, "content": msg.content or ""} 
                for msg in agui_input.messages
            ],
            "stream": True,
            "org_id": context_values.get("org_id", "default"),
            "user_id": context_values.get("user_id", "default"),
            "thread_id": agui_input.thread_id or str(uuid.uuid4()),
            "tools": tools
        }
    
    async def stream_agent_response(self, agui_input: RunAgentInput, model: str = None):
        """
        Standard streaming function that works with any agent
        """
        run_id = agui_input.run_id or str(uuid.uuid4())
        thread_id = agui_input.thread_id or str(uuid.uuid4())
        message_id = str(uuid.uuid4())
        
        try:
            # Convert to internal format
            internal_request = self.convert_agui_to_internal(agui_input, model)
            
            # Setup agent
            model_id = self.service.validate_request(internal_request)
            input_data, config = self.service.build_input_data_and_config(internal_request)
            agent_instance = self.service.setup_agent_with_tools(
                model_id, config, internal_request.get("tools", [])
            )
            agent = await agent_instance.compile(name=model_id)
            
            # Start streaming
            async def event_stream():
                try:
                    # 1. Send run start
                    yield self.encoder.encode(
                        RunStartedEvent(
                            type=EventType.RUN_STARTED,
                            run_id=run_id,
                            thread_id=thread_id
                        )
                    )
                    
                    # 2. Send initial state snapshot (if needed)
                    if agui_input.state:
                        yield self.encoder.encode(
                            StateSnapshotEvent(
                                type=EventType.STATE_SNAPSHOT,
                                snapshot=agui_input.state
                            )
                        )
                    
                    # 3. Stream agent execution
                    message_started = False
                    
                    async for stream_mode, chunk in agent.astream(
                        input_data, 
                        config=config, 
                        stream_mode=["messages", "custom", "updates"]
                    ):
                        
                        if stream_mode == "messages":
                            message_chunk, metadata = chunk
                            
                            if hasattr(message_chunk, "content") and message_chunk.content:
                                node_name = metadata.get("langgraph_node", "unknown")
                                
                                # Handle different message types based on node
                                if self._is_final_response(node_name):
                                    # Final assistant response
                                    if not message_started:
                                        yield self.encoder.encode(
                                            TextMessageStartEvent(
                                                type=EventType.TEXT_MESSAGE_START,
                                                message_id=message_id,
                                                role="assistant"
                                            )
                                        )
                                        message_started = True
                                    
                                    # Stream content
                                    yield self.encoder.encode(
                                        TextMessageContentEvent(
                                            type=EventType.TEXT_MESSAGE_CONTENT,
                                            message_id=message_id,
                                            delta=message_chunk.content
                                        )
                                    )
                                
                                elif self._is_tool_node(node_name):
                                    # Handle tool calls
                                    async for event in self._handle_tool_execution(
                                        message_chunk, metadata, message_id
                                    ):
                                        yield event
                            
                            # Handle tool calls in messages
                            if hasattr(message_chunk, "tool_calls") and message_chunk.tool_calls:
                                for tool_call in message_chunk.tool_calls:
                                    # Emit tool call events directly in the stream
                                    async for event in self._emit_tool_call_events(tool_call, message_id):
                                        yield event
                        
                        elif stream_mode == "custom":
                            # Handle custom events
                            if isinstance(chunk, dict):
                                yield self.encoder.encode(
                                    StateDeltaEvent(
                                        type=EventType.STATE_DELTA,
                                        delta=[{
                                            "op": "add",
                                            "path": "/custom_data",
                                            "value": chunk
                                        }]
                                    )
                                )
                    
                    # 4. End message if started
                    if message_started:
                        yield self.encoder.encode(
                            TextMessageEndEvent(
                                type=EventType.TEXT_MESSAGE_END,
                                message_id=message_id
                            )
                        )
                    
                    # 5. Send run completion
                    yield self.encoder.encode(
                        RunFinishedEvent(
                            type=EventType.RUN_FINISHED,
                            run_id=run_id,
                            thread_id=thread_id
                        )
                    )
                    
                except Exception as e:
                    logger.error(f"Streaming error: {e}")
                    yield self.encoder.encode(
                        RunErrorEvent(
                            type=EventType.RUN_ERROR,
                            message=str(e)
                        )
                    )
            
            return StreamingResponse(
                event_stream(), 
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                }
            )
                
        except Exception as e:
            logger.error(f"Agent setup error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _emit_tool_call_events(self, tool_call, parent_message_id):
        """Emit proper tool call event sequence"""
        tool_call_id = tool_call.id
        
        # Start tool call - using correct property name
        yield self.encoder.encode(
            ToolCallStartEvent(
                type=EventType.TOOL_CALL_START,
                tool_call_id=tool_call_id,
                tool_call_name=tool_call.function.name, 
                parent_message_id=parent_message_id     
            )
        )
        
        # Send arguments
        if tool_call.function.arguments:
            yield self.encoder.encode(
                ToolCallArgsEvent(
                    type=EventType.TOOL_CALL_ARGS,
                    tool_call_id=tool_call_id,
                    delta=tool_call.function.arguments
                )
            )
        
        # End tool call
        yield self.encoder.encode(
            ToolCallEndEvent(
                type=EventType.TOOL_CALL_END,
                tool_call_id=tool_call_id
            )
        )
    
    def _is_final_response(self, node_name: str) -> bool:
        """Determine if this is a final response node"""
        # Customize this based on your agent graph structure
        return len(str(node_name).split('$')) == 2 or node_name in ["assistant", "final", "response"]
    
    def _is_tool_node(self, node_name: str) -> bool:
        """Determine if this is a tool execution node"""
        return "tool" in str(node_name).lower()
    
    async def _handle_tool_execution(self, message_chunk, metadata, parent_message_id):
        """Handle tool execution events"""
        # Handle tool calls in intermediate tool execution steps
        if hasattr(message_chunk, "tool_calls") and message_chunk.tool_calls:
            async for event in self._emit_tool_call_events(message_chunk.tool_calls[0], parent_message_id):
                yield event


wrapper = StandardAgentWrapper()

@router.post("/copilotkit")
async def copilotkit_endpoint(request: Request):
    try:
        body = await request.json()   
             
        if "operationName" in body or "query" in body:
            return await handle_graphql_operations(body)
        
        elif "threadId" in body and "messages" in body:
            body = dict(body) 
            model = body.pop("model")
            
            # Transform tools format if needed
            if "tools" in body and body["tools"]:
                transformed_tools = []
                for tool in body["tools"]:
                    if isinstance(tool, str):
                        # Convert string to dict format
                        transformed_tools.append({
                            "name": tool,
                            "description": f"Tool: {tool}",
                            "parameters": {"type": "object", "properties": {}}
                        })
                    elif isinstance(tool, dict) and "name" in tool:
                        # Ensure required fields exist
                        transformed_tool = {
                            "name": tool["name"],
                            "description": tool.get("description", f"Tool: {tool['name']}"),
                            "parameters": tool.get("parameters", {"type": "object", "properties": {}})
                        }
                        transformed_tools.append(transformed_tool)
                    else:
                        transformed_tools.append(tool)
                body["tools"] = transformed_tools
            
            agui_input = RunAgentInput(**body)
            
            return await wrapper.stream_agent_response(agui_input, model)
        
        else:
            return await handle_regular_http_request(body)
            
    except Exception as e:
        logger.error(f"❌ Endpoint error: {e}")
        import traceback
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


async def handle_graphql_operations(body: dict):

    operation_name = body.get("operationName")
    variables = body.get("variables", {})
    
    if operation_name == "availableAgents":
        agents_list = []
        for agent in AVAILABLE_AGENTS:
            agent_info = {
                "name": agent["name"],
                "id": agent["id"],
                "description": agent["description"],
                "__typename": "Agent"
            }
            agents_list.append(agent_info)
            logger.info(f"📋 Added agent: {agent_info}")
        
        response_data = {
            "data": {
                "availableAgents": {
                    "agents": agents_list,
                    "__typename": "AgentsResponse"
                }
            }
        }
        
        logger.info(f"📋 Returning GraphQL response: {json.dumps(response_data, indent=2)}")
        return JSONResponse(content=response_data)
    
    elif operation_name == "loadAgentState":
        # Handle state loading
        data = variables.get("data", {})
        thread_id = data.get("threadId", str(uuid.uuid4()))
        agent_name = data.get("agentName", "default")
        
        # Create StandardAgentState instance
        agent_state = StandardAgentState(
            thread_id=thread_id,
            messages=[],
            tools=[],
            context={
                "agent_name": agent_name,
                "thread_id": thread_id
            }
        )
        
        response_data = {
            "data": {
                "loadAgentState": {
                    "threadId": thread_id,
                    "threadExists": False,
                    "state": agent_state.model_dump(),
                    "messages": agent_state.messages,
                    "__typename": "AgentState"
                }
            }
        }
        
        logger.info(f"📋 Load agent state response: {json.dumps(response_data, indent=2)}")
        return JSONResponse(content=response_data)
    
    elif operation_name == "generateCopilotResponse":
        try:
            data = variables.get("data", {})
            messages = variables.get("messages", [])
            agent_id = variables.get("agentId")
            thread_id = variables.get("threadId", str(uuid.uuid4()))
            
            formatted_messages = []
            for msg in messages:
                formatted_msg = {
                    "id": str(uuid.uuid4()),
                    "role": msg.get("role", "user"),
                    "content": msg.get("content") or ""
                }
                formatted_messages.append(formatted_msg)
            
            agui_request = {
                "thread_id": thread_id,
                "run_id": str(uuid.uuid4()),
                "messages": formatted_messages,
                "state": data,
                "tools": [],
                "context": [
                    {"description": "Agent ID", "value": agent_id},
                    {"description": "Thread ID", "value": thread_id}
                ],
                "forwardedProps": {}
            }
            
            agui_input = RunAgentInput(**agui_request)
            return await wrapper.stream_agent_response(agui_input, agent_id)
            
        except Exception as e:
            logger.error(f"❌ GraphQL streaming error: {e}")
            return JSONResponse(
                content={"errors": [{"message": str(e)}]},
                status_code=500
            )
    
    else:
        logger.error(f"❌ Unknown GraphQL operation: {operation_name}")
        return JSONResponse(
            content={"errors": [{"message": f"Unknown operation: {operation_name}"}]},
            status_code=400
        )


async def handle_regular_http_request(body: dict):
    messages = body.get("messages", [])
    if not messages:
        raise HTTPException(status_code=400, detail="No messages provided")
    
    formatted_messages = []
    for msg in messages:
        formatted_msg = {
            "id": str(uuid.uuid4()),  
            "role": msg.get("role", "user"),
            "content": msg.get("content") or ""
        }
        formatted_messages.append(formatted_msg)
    
    agui_request = {
        "thread_id": body.get("thread_id", str(uuid.uuid4())),
        "run_id": str(uuid.uuid4()),
        "messages": formatted_messages,  
        "state": body.get("state", {}),
        "tools": body.get("tools", []),
        "context": body.get("context", []),
        "forwardedProps": body.get("forwardedProps", {})  
    }
    
    try:
        agui_input = RunAgentInput(**agui_request)
        model = body.get("model") or body.get("agent_id")
        
        return await wrapper.stream_agent_response(agui_input, model)
    except Exception as e:
        logger.error(f"❌ RunAgentInput validation error: {e}")
        logger.error(f"❌ Request data: {agui_request}")
        raise HTTPException(status_code=400, detail=f"Invalid request format: {str(e)}")




