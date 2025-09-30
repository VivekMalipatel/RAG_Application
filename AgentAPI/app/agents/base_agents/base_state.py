from typing_extensions import TypedDict
from typing import Annotated, NotRequired, Any
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage
from langmem.short_term import RunningSummary

class BaseState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    summary_context: NotRequired[RunningSummary | dict[str, Any] | None]
    token_usage_history: NotRequired[list[dict[str, int]]]
    summarization_pending: NotRequired[bool]