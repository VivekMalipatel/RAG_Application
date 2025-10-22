from typing_extensions import TypedDict
from typing import Annotated, NotRequired, Any
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage
from langmem.short_term import RunningSummary

def add_ui_messages(left: list[dict[str, Any]] | None, right: list[dict[str, Any]] | dict[str, Any] | None) -> list[dict[str, Any]]:
    if left is None:
        left = []
    if right is None:
        return left
    
    if isinstance(right, dict):
        right = [right]
    
    if not isinstance(right, list):
        right = [right]
    
    return left + right

class BaseState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    context: NotRequired[dict[str, Any]]
    summary_context: NotRequired[RunningSummary | dict[str, Any] | None]
    token_usage_history: NotRequired[list[dict[str, int]]]
    ui: Annotated[list[dict[str, Any]], add_ui_messages]