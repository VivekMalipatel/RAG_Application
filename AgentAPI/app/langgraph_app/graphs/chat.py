import sys
from functools import lru_cache
from pathlib import Path
from typing import Literal

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from agents.base_agents.base_state import BaseState
from agents.chat_agent.chat_agent import ChatAgent
from agents.utils import _load_prompt
from langgraph_app.subgraphs.knowledge_search import (
    create_graph as create_knowledge_search_graph,
)
from langgraph_app.graphs.utils import (
    build_partitioned_config,
    flatten_content,
    messages_to_transcript,
    normalize_queries,
    parse_planner_output,
)
from llm.llm import LLM

KNOWLEDGE_SEARCH_FLAG = "enable_knowledge_search"
KNOWLEDGE_SEARCH_NAME = "knowledge_search_agent"
PLANNER_PROMPT = _load_prompt("PlannerPrompt", base_dir=Path(__file__).parent)


class PlannerDecision(BaseModel):
    action: Literal["knowledge", "respond", "end"]
    reason: str | None = None
    queries: list[str] = Field(default_factory=list)


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


def _route_from_plan(state: BaseState) -> str:
    action = state.get("next_action")
    if action == "knowledge":
        return "knowledge"
    if action == "respond":
        return "respond"
    return "end"


async def _entry_node(state: BaseState, config: RunnableConfig) -> BaseState:
    _validate_config(config)
    configurable = config.get("configurable", {})
    should_plan = bool(configurable.get(KNOWLEDGE_SEARCH_FLAG))
    route = "plan" if should_plan else "chat"
    updated = dict(state)
    updated["entry_route"] = route
    if route == "chat":
        updated["next_action"] = "respond"
        updated["knowledge_queries"] = []
        updated["plan_override"] = None
        updated["handoff_reason"] = None
    return updated


def _route_from_entry(state: BaseState) -> str:
    route = state.get("entry_route")
    if route == "chat":
        return "chat"
    return "plan"


async def _plan_node(state: BaseState, config: RunnableConfig) -> BaseState:
    _validate_config(config)
    override = state.get("plan_override")
    if override in {"respond", "end"}:
        updated = dict(state)
        updated["next_action"] = override
        updated["plan_override"] = None
        if override != "knowledge":
            updated["knowledge_queries"] = []
            updated["handoff_reason"] = None
        return updated
    configurable = config.get("configurable", {})
    if not configurable.get(KNOWLEDGE_SEARCH_FLAG):
        updated = dict(state)
        updated["next_action"] = "respond"
        updated["knowledge_queries"] = []
        updated["handoff_reason"] = None
        updated["plan_override"] = None
        return updated
    messages = state.get("messages", [])
    if not messages:
        updated = dict(state)
        updated["next_action"] = "respond"
        updated["knowledge_queries"] = []
        updated["handoff_reason"] = None
        updated["plan_override"] = None
        return updated
    transcript = messages_to_transcript(messages)
    if not transcript:
        updated = dict(state)
        updated["next_action"] = "respond"
        updated["knowledge_queries"] = []
        updated["handoff_reason"] = None
        updated["plan_override"] = None
        return updated
    planner_llm = LLM()
    planner_messages = [
        SystemMessage(content=PLANNER_PROMPT),
        HumanMessage(content=transcript),
    ]
    planner_response = await planner_llm.ainvoke(planner_messages, config=config)
    raw_content = getattr(planner_response, "content", planner_response)
    parsed_decision = parse_planner_output(flatten_content(raw_content), PlannerDecision)
    decision = parsed_decision
    action = decision.action
    queries = normalize_queries(decision.queries)
    if action == "knowledge" and not queries:
        action = "respond"
    if action != "knowledge":
        queries = []
    updated = dict(state)
    updated["next_action"] = action
    updated["knowledge_queries"] = queries
    updated["handoff_reason"] = decision.reason
    updated["plan_override"] = None
    return updated


async def _knowledge_search_node(state: BaseState, config: RunnableConfig) -> BaseState:
    _validate_config(config)
    configurable = config.get("configurable", {})
    if not configurable.get(KNOWLEDGE_SEARCH_FLAG):
        updated = dict(state)
        updated["knowledge_queries"] = []
        updated["plan_override"] = "respond"
        return updated
    queries = list(state.get("knowledge_queries") or [])
    if not queries:
        updated = dict(state)
        updated["plan_override"] = "respond"
        return updated
    knowledge_graph = create_knowledge_search_graph()
    derived_config = build_partitioned_config(config, KNOWLEDGE_SEARCH_NAME)
    merged_messages = list(state.get("messages", []))
    for query in queries:
        prompt_message = HumanMessage(content=query)
        working_state = {**state, "messages": merged_messages + [prompt_message]}
        knowledge_state = await knowledge_graph.ainvoke(working_state, config=derived_config)
        knowledge_messages = knowledge_state.get("messages", []) if knowledge_state else []
        if not knowledge_messages:
            continue
        knowledge_message = knowledge_messages[-1]
        merged_messages.append(knowledge_message)
    updated = dict(state)
    updated["messages"] = merged_messages
    updated["knowledge_queries"] = []
    updated["plan_override"] = "respond"
    return updated


async def _chat_node(state: BaseState, config: RunnableConfig) -> BaseState:
    _validate_config(config)
    agent = ChatAgent(
        config=config,
        model_kwargs={},
        vlm_kwargs={},
        node_kwargs={},
    )
    compiled_agent = await agent.compile(name="chat_agent")
    input_payload = {
        "messages": state.get("messages", []),
        "user_id": config["configurable"]["user_id"],
        "org_id": config["configurable"]["org_id"],
    }
    result = await compiled_agent.ainvoke(input_payload, config=config)
    updated: BaseState = {**state, **result}
    updated["knowledge_queries"] = []
    updated["next_action"] = "end"
    updated["plan_override"] = None
    updated["handoff_reason"] = None
    return updated


@lru_cache
def create_graph() -> CompiledStateGraph:
    builder = StateGraph(BaseState)
    builder.add_node("entry", _entry_node)
    builder.add_node("plan", _plan_node)
    builder.add_node("knowledge_search", _knowledge_search_node)
    builder.add_node("chat", _chat_node)
    builder.add_edge(START, "entry")
    builder.add_conditional_edges(
        "entry",
        _route_from_entry,
        {
            "plan": "plan",
            "chat": "chat",
        },
    )
    builder.add_conditional_edges(
        "plan",
        _route_from_plan,
        {
            "knowledge": "knowledge_search",
            "respond": "chat",
            "end": END,
        },
    )
    builder.add_edge("knowledge_search", "plan")
    builder.add_edge("chat", END)
    return builder.compile()
