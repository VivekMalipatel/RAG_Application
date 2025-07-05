from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import uuid
import logging

from app.agents.base_agent import BaseAgent
from app.agents.base_types.single_shot import SingleShotAgent
from app.agents.base_types.deep_research import DeepResearchAgent

logger = logging.getLogger(__name__)

router = APIRouter()

class AgentRequest(BaseModel):
    agent_type: str
    query: str
    thread_id: Optional[str] = None
    config: Optional[Dict[str, Any]] = None

class AgentResponse(BaseModel):
    response: str
    thread_id: str
    agent_type: str
    status: str

class AgentCreateRequest(BaseModel):
    name: str
    agent_type: str
    role: str
    instructions: str
    capabilities: List[str] = []

class AgentInfo(BaseModel):
    id: str
    name: str
    agent_type: str
    role: str
    instructions: str
    capabilities: List[str]
    status: str

# In-memory storage for agents (replace with proper storage later)
agents_store: Dict[str, Dict[str, Any]] = {}

@router.post("/create", response_model=AgentInfo)
async def create_agent(request: AgentCreateRequest):
    """Create a new agent"""
    try:
        agent_id = str(uuid.uuid4())
        
        agent_data = {
            "id": agent_id,
            "name": request.name,
            "agent_type": request.agent_type,
            "role": request.role,
            "instructions": request.instructions,
            "capabilities": request.capabilities,
            "status": "active"
        }
        
        agents_store[agent_id] = agent_data
        
        logger.info(f"Created agent {request.name} with ID {agent_id}")
        
        return AgentInfo(**agent_data)
        
    except Exception as e:
        logger.error(f"Error creating agent: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create agent: {str(e)}")

@router.get("/list", response_model=List[AgentInfo])
async def list_agents():
    """List all agents"""
    try:
        agents = [AgentInfo(**agent) for agent in agents_store.values()]
        return agents
    except Exception as e:
        logger.error(f"Error listing agents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list agents: {str(e)}")

@router.get("/{agent_id}", response_model=AgentInfo)
async def get_agent(agent_id: str):
    """Get agent by ID"""
    try:
        if agent_id not in agents_store:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        return AgentInfo(**agents_store[agent_id])
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting agent {agent_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get agent: {str(e)}")

@router.delete("/{agent_id}")
async def delete_agent(agent_id: str):
    """Delete an agent"""
    try:
        if agent_id not in agents_store:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        del agents_store[agent_id]
        logger.info(f"Deleted agent {agent_id}")
        
        return {"message": "Agent deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting agent {agent_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete agent: {str(e)}")

@router.post("/execute", response_model=AgentResponse)
async def execute_agent(request: AgentRequest, background_tasks: BackgroundTasks):
    """Execute an agent with a query"""
    try:
        thread_id = request.thread_id or str(uuid.uuid4())
        
        # Get the appropriate agent class based on type
        agent_class = None
        if request.agent_type == "single_shot":
            agent_class = SingleShotAgent
        elif request.agent_type == "deep_research":
            agent_class = DeepResearchAgent
        else:
            raise HTTPException(status_code=400, detail=f"Unknown agent type: {request.agent_type}. Available types: single_shot, deep_research")
        
        # Initialize and execute agent
        agent = agent_class()
        config = request.config or {}
        config["thread_id"] = thread_id
        
        result = await agent.execute(request.query, config)
        
        response = AgentResponse(
            response=result.get("response", ""),
            thread_id=thread_id,
            agent_type=request.agent_type,
            status="completed"
        )
        
        logger.info(f"Agent {request.agent_type} executed successfully for thread {thread_id}")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing agent: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to execute agent: {str(e)}")

@router.post("/{agent_id}/chat", response_model=AgentResponse)
async def chat_with_agent(agent_id: str, request: AgentRequest):
    """Chat with a specific agent"""
    try:
        if agent_id not in agents_store:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        agent_data = agents_store[agent_id]
        thread_id = request.thread_id or str(uuid.uuid4())
        
        # Get the appropriate agent class based on stored type
        agent_class = None
        if agent_data["agent_type"] == "single_shot":
            agent_class = SingleShotAgent
        elif agent_data["agent_type"] == "deep_research":
            agent_class = DeepResearchAgent
        else:
            raise HTTPException(status_code=400, detail=f"Unknown agent type: {agent_data['agent_type']}. Available types: single_shot, deep_research")
        
        # Initialize agent with stored configuration
        agent = agent_class(
            role=agent_data["role"],
            instructions=agent_data["instructions"]
        )
        
        config = request.config or {}
        config["thread_id"] = thread_id
        config["agent_id"] = agent_id
        
        result = await agent.execute(request.query, config)
        
        response = AgentResponse(
            response=result.get("response", ""),
            thread_id=thread_id,
            agent_type=agent_data["agent_type"],
            status="completed"
        )
        
        logger.info(f"Chat with agent {agent_id} completed for thread {thread_id}")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error chatting with agent {agent_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to chat with agent: {str(e)}")

@router.get("/{agent_id}/threads")
async def get_agent_threads(agent_id: str):
    """Get all chat threads for an agent"""
    try:
        if agent_id not in agents_store:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # This would normally fetch from storage
        # For now, return empty list as we're using in-memory storage
        return {"agent_id": agent_id, "threads": []}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting threads for agent {agent_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get agent threads: {str(e)}")

@router.get("/{agent_id}/threads/{thread_id}/history")
async def get_thread_history(agent_id: str, thread_id: str):
    """Get chat history for a specific thread"""
    try:
        if agent_id not in agents_store:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # This would normally fetch from storage
        # For now, return empty history as we're using in-memory storage
        return {
            "agent_id": agent_id,
            "thread_id": thread_id,
            "history": []
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting thread history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get thread history: {str(e)}")
