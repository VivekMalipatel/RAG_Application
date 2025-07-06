from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
import typing
from typing import Any, Sequence, Union, Optional, Callable
from langchain_core.tools import BaseTool

from app.llm.llm import LLM
from app.agents.base_states.simple_state import State
from app.agents.base_checkpointers.simple_checkpointer import MemorySaver

graph_builder = StateGraph(State)

class SingleShotAgent:

    def __init__(self, prompt=None):

        self.llm = LLM()
        self.graph_builder = StateGraph(State)
        self.checkpointer = MemorySaver()
        self.compiled_graph : CompiledStateGraph = None

        self.prompt = prompt

    def bind_tools(self,
        tools: Sequence[
            Union[typing.Dict[str, Any], type, Callable, BaseTool]
        ],
        *,
        tool_choice: Optional[Union[str]] = None,
        **kwargs: Any,
        ):
        self.tools = tools
        self.reasoning_llm_with_tools = self.llm.bind_tools(
            tools, 
            tool_choice=tool_choice, 
            **kwargs
        )
        self.compiled_graph = self.build_graph_with_tools() if tools else self.compiled_graph
        return self

    async def llm_node(self, state: State):
        messages = state["messages"]
        if self.prompt:
            messages.insert(0, {"role": "system", "content": self.prompt})

        response = await self.llm.ainvoke(messages)
        return {"messages": [response]}
    
    def build_graph_without_tools(self):
        self.graph_builder.add_node("llm", self.llm_node)
        self.graph_builder.add_edge(START, "llm")
        self.graph_builder.add_edge("llm", END)

        self.compiled_graph = self.graph_builder.compile(checkpointer=self.checkpointer)
        return self

    def build_graph_with_tools(self):
        self.graph_builder.add_node("llm", self.llm_node)
        self.graph_builder.add_node("tools", ToolNode(self.tools))
        
        self.graph_builder.add_edge(START, "llm")
        self.graph_builder.add_conditional_edges(
            "llm",
            tools_condition,
            {"tools": "tools", "__end__": END}
        )
        self.graph_builder.add_edge("tools", "llm")

        self.compiled_graph = self.graph_builder.compile(checkpointer=self.checkpointer)
        return self

