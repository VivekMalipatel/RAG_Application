from fastapi import APIRouter
from agents import AVAILABLE_AGENTS

router = APIRouter()

@router.get("/v1/models")
async def openai_models():
    models = []
    for agent in AVAILABLE_AGENTS:
        models.append({
            "id": agent.get("id"),
            "object": "model",
            "created": agent.get("created"),
            "owned_by": "openai",
            "root": None,
            "parent": None,
            "permission": [],
            "tags": agent.get("tags", []),
            "name": agent.get("name"),
            "actions": agent.get("actions", []),
            "filters": agent.get("filters", []),
            "connection_type": agent.get("connection_type", "external"),
            "openai": {
                "id": agent.get("id"),
                "object": "model",
                "created": agent.get("created"),
                "owned_by": "openai",
                "root": None,
                "parent": None,
            },
            "urlIdx": 0,
        })
    return {"object": "list", "data": models}

