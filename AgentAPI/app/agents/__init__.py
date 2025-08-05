from agents.chat_agent.chat_agent import ChatAgent
from agents.deep_research.deep_research import DeepResearchAgent
from agents.waiter_agent.waiter import WaiterAgent
import time

AVAILABLE_AGENTS = [
    {
        "id": "chat_agent",
        "name": "ChatAgent",
        "description": "A general-purpose chat agent for conversational interactions",
        "created": int(time.time())
    },
    {
        "id": "deep_research_agent",
        "name": "DeepResearchAgent",
        "description": "An agent specialized in deep research tasks",
        "created": int(time.time())
    },
    {
        "id": "waiter_agent",
        "name": "WaiterAgent",
        "description": "An agent specialized in restaurant service tasks",
        "created": int(time.time())
    }
]

AGENT_CLASS_MAP = {
    "chat_agent": ChatAgent,
    "deep_research_agent": DeepResearchAgent,
    "waiter_agent": WaiterAgent
}

def get_agent_by_id(agent_id: str):
    return AGENT_CLASS_MAP.get(agent_id)
