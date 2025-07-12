from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from agents import AVAILABLE_AGENTS
import time

router = APIRouter()

class Message(BaseModel):
    role: str
    content: str

class AgentCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Optional[Dict[str, int]] = None

@router.get("/v1/models")
async def openai_models():
    models = []
    for agent in AVAILABLE_AGENTS:
        model_id = agent.get("id")
        name = agent.get("name")
        created = int(time.time())
        models.append({
            "id": model_id,
            "object": "model",
            "created": created,
            "owned_by": agent.get("owned_by", "openai"),
            "root": None,
            "parent": None,
            "permission": [],
            "tags": agent.get("tags", []),
            "name": name,
            "actions": agent.get("actions", []),
            "filters": agent.get("filters", []),
            "connection_type": agent.get("connection_type", "external"),
            "openai": {
                "id": model_id,
                "object": "model",
                "created": created,
                "owned_by": agent.get("owned_by", "openai"),
                "root": None,
                "parent": None,
            },
            "urlIdx": 0,
        })
    return {"object": "list", "data": models}

@router.get("/v1/api/models")
async def list_agents():
    """List all available agents with id, name, and description."""
    return {"agents": AVAILABLE_AGENTS}
