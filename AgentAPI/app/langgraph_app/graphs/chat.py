import sys
from functools import lru_cache
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.redis import RedisSaver

from agents.base_agents.base_state import BaseState
from agents.chat_agent.chat_agent import ChatAgent
from tools.agents_as_tools.knowledge_search.knowledge_search import (
    knowledge_search_agent as knowledge_search_tool,
)
from config import config as app_config

KNOWLEDGE_SEARCH_FLAG = "enable_knowledge_search"

def _validate_config(config: RunnableConfig) -> None:
    if not config:
        raise ValueError("Config required")
    configurable = config.get("configurable")
    if not configurable:
        raise ValueError("Configurable values required")
    if not configurable.get("user_id"):
        raise ValueError("user_id required")
    if not configurable.get("org_id"):
        raise ValueError("org_id required")
    if not configurable.get("thread_id"):
        raise ValueError("thread_id required")

async def _entry_node(state: BaseState, config: RunnableConfig) -> BaseState:
    _validate_config(config)
    return dict(state)


async def _chat_node(state: BaseState, config: RunnableConfig) -> BaseState:
    _validate_config(config)
    configurable = config.get("configurable", {})
    knowledge_enabled = bool(configurable.get(KNOWLEDGE_SEARCH_FLAG))

    agent = ChatAgent(
        config=config,
        model_kwargs={},
        vlm_kwargs={},
        node_kwargs={},
    )
    if knowledge_enabled:
        agent = agent.bind_tools([knowledge_search_tool])
    compiled_agent = await agent.compile(name="chat_agent")
    input_payload = {
        "messages": state.get("messages", []),
        "user_id": configurable["user_id"],
        "org_id": configurable["org_id"],
    }
    result = await compiled_agent.ainvoke(input_payload, config=config)
    updated: BaseState = {**state, **result}
    return updated


@lru_cache
def create_graph() -> CompiledStateGraph:
    builder = StateGraph(BaseState)
    builder.add_node("entry", _entry_node)
    builder.add_node("chat", _chat_node)
    builder.add_edge(START, "entry")
    builder.add_edge("entry", "chat")
    builder.add_edge("chat", END)
    checkpointer = RedisSaver.from_conn_string(app_config.REDIS_URI)
    return builder.compile(checkpointer=checkpointer)
