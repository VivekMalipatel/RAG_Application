from fastapi import APIRouter

from backend.agents.catalog import get_agent_catalog
from backend.agents.schema import AgentDefinition

router = APIRouter(prefix="/agents", tags=["agents"])


@router.get("/catalog", response_model=list[AgentDefinition])
async def read_agent_catalog() -> list[AgentDefinition]:
    return get_agent_catalog()
