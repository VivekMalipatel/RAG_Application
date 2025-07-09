from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph.message import add_messages


class BaseState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    org_id: str