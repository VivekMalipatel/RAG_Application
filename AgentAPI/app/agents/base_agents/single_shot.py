from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore
from langgraph.types import All
import typing
from typing import Any, Sequence, Union, Optional, Callable, AsyncIterator
from langchain_core.tools import BaseTool
from langchain_core.runnables import RunnableConfig, Runnable
from langchain_core.language_models.chat_models import LanguageModelInput
from langchain_core.runnables.schema import StreamEvent
from langchain_core.runnables.utils import Output
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.base import CheckpointTuple
from openai import BaseModel

from llm.llm import LLM
from agents.base_states.simple_state import State
from agents.base_checkpointers.simple_checkpointer import MemorySaver


class SingleShotAgent:

    def __init__(self, 
                 prompt: Optional[str] = None, 
                 config: Optional[RunnableConfig] = None,
                 **kwargs: Any):

        self.llm = LLM()
        self.graph_builder = StateGraph(State)
        self.checkpointer = MemorySaver()
        self.compiled_graph: CompiledStateGraph = None
        self.tools: Sequence[Union[typing.Dict[str, Any], type, Callable, BaseTool]] = []

        self.prompt = prompt
        self.config = config
        self.llm_kwargs = kwargs

    def set_prompt(self, prompt: str) -> 'SingleShotAgent':
        self.prompt = prompt
        return self

    def set_config(self, config: RunnableConfig) -> 'SingleShotAgent':
        self.config = config
        return self

    def set_llm_kwargs(self, **kwargs: Any) -> 'SingleShotAgent':
        self.llm_kwargs.update(kwargs)
        return self

    def _rebuild_graph(self) -> None:
        if self.tools:
            self.compile_with_tools()
        else:
            self.compile_without_tools()
    
    def with_structured_output(
        self, schema: Union[dict, type[BaseModel]], **kwargs: Any
    ) -> 'SingleShotAgent':
        self.llm:LLM = self.llm.with_structured_output(
            schema=schema, 
            **kwargs
        
        )
        return self

    def bind_tools(self,
                    tools: Sequence[
                        Union[typing.Dict[str, Any], type, Callable, BaseTool]
                    ],
                    *,
                    tool_choice: Optional[Union[str]] = None, 
                    **kwargs: Any,
                 ) -> 'SingleShotAgent':
        self.tools = tools
        self.llm = self.llm.bind_tools(
            tools, 
            tool_choice=tool_choice, 
            **kwargs
        )
        self._rebuild_graph()
        return self

    async def llm_node(self, state: State):
        messages = state["messages"]
        if self.prompt:
            messages.insert(0, {"role": "system", "content": self.prompt})

        async for chunk in self.llm.astream(messages, config=self.config, **self.llm_kwargs):
            yield chunk
    
    def compile_without_tools(self,
                        checkpointer: Optional[BaseCheckpointSaver] = None,
                        *,
                        store: Optional[BaseStore] = None,
                        interrupt_before: Optional[Union[All, list[str]]] = None,
                        interrupt_after: Optional[Union[All, list[str]]] = None,
                        debug: bool = False,
                        name: Optional[str] = None,
                    ) -> 'SingleShotAgent':
        
        self.graph_builder = StateGraph(State)
        
        self.graph_builder.add_node("llm", self.llm_node)
        self.graph_builder.add_edge(START, "llm")
        self.graph_builder.add_edge("llm", END)

        compile_checkpointer = checkpointer if checkpointer is not None else self.checkpointer
        self.compiled_graph = self.graph_builder.compile(
            checkpointer=compile_checkpointer,
            store=store,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
            debug=debug,
            name=name
        )
        return self

    def compile_with_tools(self,
                        checkpointer: Optional[BaseCheckpointSaver] = None,
                        *,
                        store: Optional[BaseStore] = None,
                        interrupt_before: Optional[Union[All, list[str]]] = None,
                        interrupt_after: Optional[Union[All, list[str]]] = None,
                        debug: bool = False,
                        name: Optional[str] = None,
                    ) -> 'SingleShotAgent':
        
        self.graph_builder = StateGraph(State)
        
        self.graph_builder.add_node("llm", self.llm_node)
        self.graph_builder.add_node("tools", ToolNode(self.tools))
        
        self.graph_builder.add_edge(START, "llm")
        self.graph_builder.add_conditional_edges(
            "llm",
            tools_condition,
            {"tools": "tools", "__end__": END}
        )
        self.graph_builder.add_edge("tools", "llm")

        compile_checkpointer = checkpointer if checkpointer is not None else self.checkpointer
        self.compiled_graph = self.graph_builder.compile(
            checkpointer=compile_checkpointer,
            store=store,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
            debug=debug,
            name=name
        )
        return self

    def compile(self,
               checkpointer: Optional[BaseCheckpointSaver] = None,
               *,
               store: Optional[BaseStore] = None,
               interrupt_before: Optional[Union[All, list[str]]] = None,
               interrupt_after: Optional[Union[All, list[str]]] = None,
               debug: bool = False,
               name: Optional[str] = None,
               ) -> 'SingleShotAgent':
        if self.tools:
            return self.compile_with_tools(
                checkpointer=checkpointer,
                store=store,
                interrupt_before=interrupt_before,
                interrupt_after=interrupt_after,
                debug=debug,
                name=name
            )
        else:
            return self.compile_without_tools(
                checkpointer=checkpointer,
                store=store,
                interrupt_before=interrupt_before,
                interrupt_after=interrupt_after,
                debug=debug,
                name=name
            )

    async def astream_events(self, messages: list, **kwargs: Any) -> AsyncIterator[StreamEvent]:
        if not self.compiled_graph:
            self.compile()
        
        state = {"messages": messages}
        async for event in self.compiled_graph.astream_events(state, config=self.config, **kwargs):
            yield event

