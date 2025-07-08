import os
import importlib
import inspect
from agents.base_agents.base_agent import BaseAgent

AGENTS_DIR = os.path.dirname(__file__)
AVAILABLE_AGENTS = []
AGENT_CLASS_MAP = {}

for entry in os.listdir(AGENTS_DIR):
    entry_path = os.path.join(AGENTS_DIR, entry)
    if (
        os.path.isdir(entry_path)
        and not entry.startswith("__")
        and os.path.exists(os.path.join(entry_path, "__init__.py"))
    ):
        module_name = f"agents.{entry}"
        try:
            module = importlib.import_module(module_name)
            for name, obj in inspect.getmembers(module):
                if (
                    inspect.isclass(obj)
                    and issubclass(obj, BaseAgent)
                    and obj is not BaseAgent
                ):
                    agent_id = entry
                    AVAILABLE_AGENTS.append({
                        "id": agent_id,
                        "name": name,
                        "description": getattr(obj, "__doc__", "") or f"{name} agent"
                    })
                    AGENT_CLASS_MAP[agent_id] = obj
        except Exception as e:
            print(f"Could not import agent from {module_name}: {e}")

def get_agent_by_id(agent_id: str):
    """Return the agent class for a given id, or None if not found."""
    return AGENT_CLASS_MAP.get(agent_id)
