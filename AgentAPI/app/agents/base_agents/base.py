from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore
from langgraph.types import All
import typing
import asyncio
from typing import Any, Sequence, Union, Optional, Callable, AsyncIterator
from langchain_core.tools import BaseTool
from langchain_core.runnables import RunnableConfig, Runnable
from langchain_core.language_models.chat_models import LanguageModelInput
from langchain_core.runnables.schema import StreamEvent
from langchain_core.runnables.utils import Output
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.base import CheckpointTuple
from openai import BaseModel

from llm.llm import LLM
from agents.base_states.simple_state import State
from agents.base_checkpointers.simple_checkpointer import MemorySaver

import asyncio
import logging


class BaseAgent:

    def __init__(self, 
                 config: Optional[RunnableConfig] = None,
                 **kwargs: Any):

        self.llm = LLM()
        self.graph_builder = StateGraph(State)
        self.compiled_graph: CompiledStateGraph = None
        self.tools: Sequence[Union[typing.Dict[str, Any], type, Callable, BaseTool]] = []

        self.config = config
        self.llm_kwargs = kwargs

    def set_prompt(self, prompt: str) -> 'BaseAgent':
        self.prompt = prompt
        return self

    def set_config(self, config: RunnableConfig) -> 'BaseAgent':
        self.config = config
        return self

    def set_llm_kwargs(self, **kwargs: Any) -> 'BaseAgent':
        self.llm_kwargs.update(kwargs)
        return self

    def _rebuild_graph(self) -> None:
        self.compiled_graph = None
        if self.tools:
            self.compile_with_tools()
        else:
            self.compile_without_tools()
    
    def with_structured_output(
        self, schema: Union[dict, type[BaseModel]], **kwargs: Any
    ) -> 'BaseAgent':
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
                 ) -> 'BaseAgent':
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

        response = await self.llm.ainvoke(messages, config=self.config, **self.llm_kwargs)
        return {"messages": [response]}
    
    def compile_without_tools(self,
                        checkpointer: Optional[BaseCheckpointSaver] = None,
                        *,
                        store: Optional[BaseStore] = None,
                        interrupt_before: Optional[Union[All, list[str]]] = None,
                        interrupt_after: Optional[Union[All, list[str]]] = None,
                        debug: bool = False,
                        name: Optional[str] = None,
                    ) -> 'BaseAgent':
        
        self.graph_builder = StateGraph(State)
        
        self.graph_builder.add_node("llm", self.llm_node)
        self.graph_builder.add_edge(START, "llm")
        self.graph_builder.add_edge("llm", END)

        self.compiled_graph = self.graph_builder.compile(
            checkpointer=checkpointer,
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
                    ) -> 'BaseAgent':
        
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

        self.compiled_graph = self.graph_builder.compile(
            checkpointer=checkpointer,
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
               ) -> 'BaseAgent':
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
            logging.error("Compiled graph is not initialized. Please call compile() before streaming events.")
            raise RuntimeError("Compiled graph is not initialized. Please call compile() before streaming events.")
        
        state = {"messages": messages}
        async for event in self.compiled_graph.astream_events(state, config=self.config, **kwargs):
            yield event

if __name__ == "__main__":
    async def test_base_agent():

        test_agent = BaseAgent(
            config={"configurable": {"thread_id": "test_thread"}}
        )

        test_agent.compile(
            checkpointer=MemorySaver(),
            store=None,
            interrupt_before=None,
            interrupt_after=None,
            debug=False,
            name="test_single_shot_agent"
        )
        
        output_lines = []
        
        output_lines.append("=== Multi-round Image Conversation ===")
        
        try:
            output_lines.append("\n--- Round 1: Initial image question ---")
            round1_messages = [SystemMessage(content="You are a helpful assistant with vision capabilities."),
                HumanMessage(content=[
                    {"type": "text", "text": "What do you see in this image?"},
                    {"type": "image_url", "image_url": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"}}
                ])
            ]
            async for event in test_agent.astream_events(round1_messages):
                print(event)
                output_lines.append(f"Event: {event}")
            
            output_lines.append("\n--- Round 2: Follow-up about colors ---")
            round2_messages = [HumanMessage(content="Can you describe the colors in more detail?")]
            async for event in test_agent.astream_events(round2_messages):
                print(event)
                output_lines.append(f"Event: {event}")
            
            output_lines.append("\n--- Round 3: Activities question ---")
            round3_messages = [HumanMessage(content="What kind of activities could someone do in this location?")]
            async for event in test_agent.astream_events(round3_messages):
                print(event)
                output_lines.append(f"Event: {event}")
            
            output_lines.append("\n--- Round 4: Photography question ---")
            round4_messages = [HumanMessage(content="Is this a good place for photography? Why?")]
            async for event in test_agent.astream_events(round4_messages):
                print(event)
                output_lines.append(f"Event: {event}")
                
        except Exception as e:
            output_lines.append(f"Error: {e}")
        
        with open("single_shot_output.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(output_lines))
        
        print(f"\nOutput saved to single_shot_output.txt ({len(output_lines)} lines)")

    asyncio.run(test_base_agent())

