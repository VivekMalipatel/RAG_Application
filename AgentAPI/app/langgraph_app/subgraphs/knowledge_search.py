import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langchain_core.runnables import RunnableConfig

from agents.base_agents.base_state import BaseState
from agents.util_agents.knowledge_search_agent.knowledge_search_agent import (
    KnowledgeSearchAgent,
)


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


async def _knowledge_search_node(state: BaseState, config: RunnableConfig) -> dict[str, Any]:
    _validate_config(config)
    agent = KnowledgeSearchAgent(
        config=config,
        model_kwargs={},
        vlm_kwargs={},
        node_kwargs={},
    )
    compiled_agent = await agent.compile(name="knowledge_search_agent")
    input_payload = {
        "messages": state.get("messages", []),
        "user_id": config["configurable"]["user_id"],
        "org_id": config["configurable"]["org_id"],
    }
    return await compiled_agent.ainvoke(input_payload, config=config)


@lru_cache
def create_graph() -> CompiledStateGraph:
    builder = StateGraph(BaseState)
    builder.add_node("knowledge_search", _knowledge_search_node)
    builder.add_edge(START, "knowledge_search")
    builder.add_edge("knowledge_search", END)
    return builder.compile(checkpointer=False)
