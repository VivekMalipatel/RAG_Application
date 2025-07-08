
from .base_agents.base import BaseAgent
from .base_agents.deep_research import DeepResearchAgent

AVAILABLE_AGENTS = [
    {"id": "base", "name": "BaseAgent", "description": "Basic research/chat agent."},
    {"id": "deep_research", "name": "DeepResearchAgent", "description": "Multi-step deep research agent."},
]

# Map agent id to class for instantiation
AGENT_CLASS_MAP = {
    "base": BaseAgent,
    "deep_research": DeepResearchAgent,
    # Add more mappings as you add agents
}

def get_agent_by_id(agent_id: str):
    """Return the agent class for a given id, or None if not found."""
    return AGENT_CLASS_MAP.get(agent_id)
