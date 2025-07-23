from agents.chat_agent.chat_agent import ChatAgent
import time

AVAILABLE_AGENTS = [
    {
        "id": "chat_agent",
        "name": "ChatAgent",
        "description": "A general-purpose chat agent for conversational interactions",
        "created": int(time.time())
    }
]

AGENT_CLASS_MAP = {
    "chat_agent": ChatAgent
}

def get_agent_by_id(agent_id: str):
    return AGENT_CLASS_MAP.get(agent_id)
