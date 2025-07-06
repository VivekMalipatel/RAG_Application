from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore
from langgraph.types import All, StreamMode
import typing
import asyncio
from typing import Any, Sequence, Union, Optional, Callable, AsyncIterator, Literal
from langchain_core.tools import BaseTool
from langchain_core.runnables import RunnableConfig, Runnable
from langchain_core.language_models.chat_models import LanguageModelInput
from langchain_core.runnables.schema import StreamEvent
from langchain_core.runnables.utils import Output
from langchain_core.runnables.base import Input
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.base import CheckpointTuple
from openai import BaseModel
from dataclasses import dataclass, field
from functools import wraps
from copy import deepcopy

from llm.llm import LLM
from agents.base_states.simple_state import State
from agents.base_checkpointers.simple_checkpointer import MemorySaver

import asyncio
import logging


@dataclass(frozen=True)
class AgentConfig:
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    vlm_kwargs: dict[str, Any] = field(default_factory=dict)
    node_kwargs: dict[str, Any] = field(default_factory=dict)
    debug: bool = False


def requires_compile(fn):
    @wraps(fn)
    async def wrapper(self, *args, **kwargs):
        if not self._compiled_graph:
            raise ValueError("Agent not compiled. Call compile() first.")
        return await fn(self, *args, **kwargs)
    return wrapper


class BaseAgent:

    def __init__(self,
                 *,
                 model_kwargs: Optional[dict[str, Any]] = None,
                 vlm_kwargs: Optional[dict[str, Any]] = None,
                 node_kwargs: Optional[dict[str, Any]] = None,
                 debug: bool = False):
        
        self._config = AgentConfig(
            model_kwargs=model_kwargs or {},
            vlm_kwargs=vlm_kwargs or {},
            node_kwargs=node_kwargs or {},
            debug=debug
        )
        self._compiled_graph: Optional[CompiledStateGraph] = None
        self._lock = asyncio.Lock()
        self._llm = LLM(self._config.model_kwargs, self._config.vlm_kwargs)
        self._tools: Sequence[Union[typing.Dict[str, Any], type, Callable, BaseTool]] = []
        self._logger = logging.getLogger(__name__)
        
        self._checkpointer = None
        self._store = None
        self._interrupt_before = None
        self._interrupt_after = None
        self._name = None

    def _validate_tools(self, tools: Sequence[Union[typing.Dict[str, Any], type, Callable, BaseTool]]):
        for tool in tools:
            if hasattr(tool, 'run') or hasattr(tool, 'arun') or isinstance(tool, BaseTool):
                continue
            elif callable(tool):
                continue
            elif isinstance(tool, dict):
                continue
            else:
                raise ValueError(f"Invalid tool: {tool}. Tools must be callable, BaseTool instances, or dictionaries.")

    def with_structured_output(
        self, schema: Union[dict, type[BaseModel]], **kwargs: Any
    ) -> 'BaseAgent':
        new_agent = self._clone()
        new_agent._llm = new_agent._llm.with_structured_output(schema=schema, **kwargs)
        return new_agent

    def bind_tools(self,
                   tools: Sequence[Union[typing.Dict[str, Any], type, Callable, BaseTool]],
                   *,
                   tool_choice: Optional[Union[str]] = None,
                   **kwargs: Any) -> 'BaseAgent':
        
        self._validate_tools(tools)
        new_agent = self._clone()
        new_agent._tools = tools
        new_agent._llm = new_agent._llm.bind_tools(tools, tool_choice=tool_choice, **kwargs)
        return new_agent

    def _clone(self) -> 'BaseAgent':
        new_agent = BaseAgent(
            model_kwargs=self._config.model_kwargs,
            vlm_kwargs=self._config.vlm_kwargs,
            node_kwargs=self._config.node_kwargs,
            debug=self._config.debug
        )
        new_agent._tools = self._tools
        new_agent._checkpointer = self._checkpointer
        new_agent._store = self._store
        new_agent._interrupt_before = self._interrupt_before
        new_agent._interrupt_after = self._interrupt_after
        new_agent._name = self._name
        return new_agent

    async def llm_node(self, state: State):
        messages = state["messages"]
        try:
            response = await self._llm.ainvoke(messages, **self._config.node_kwargs)
            return {"messages": [response]}
        except Exception as e:
            self._logger.error(f"Error in llm_node: {e}")
            raise

    def _compile_graph(self, has_tools: bool, **compile_kwargs) -> CompiledStateGraph:
        graph_builder = StateGraph(State)
        
        graph_builder.add_node("llm", self.llm_node)
        
        if has_tools:
            graph_builder.add_node("tools", ToolNode(self._tools))
            graph_builder.add_edge(START, "llm")
            graph_builder.add_conditional_edges(
                "llm",
                tools_condition,
                {"tools": "tools", "__end__": END}
            )
            graph_builder.add_edge("tools", "llm")
        else:
            graph_builder.add_edge(START, "llm")
            graph_builder.add_edge("llm", END)

        return graph_builder.compile(**compile_kwargs)

    def compile(self,
                checkpointer: Optional[BaseCheckpointSaver] = None,
                *,
                store: Optional[BaseStore] = None,
                interrupt_before: Optional[Union[All, list[str]]] = None,
                interrupt_after: Optional[Union[All, list[str]]] = None,
                debug: Optional[bool] = None,
                name: Optional[str] = None) -> 'BaseAgent':

        self._checkpointer = checkpointer
        self._store = store
        self._interrupt_before = interrupt_before
        self._interrupt_after = interrupt_after
        self._name = name

        compile_kwargs = {
            "checkpointer": checkpointer,
            "store": store,
            "interrupt_before": interrupt_before,
            "interrupt_after": interrupt_after,
            "debug": debug if debug is not None else self._config.debug,
            "name": name
        }

        self._compiled_graph = self._compile_graph(bool(self._tools), **compile_kwargs)
        return self

    @requires_compile
    async def ainvoke(self,
                      input: dict[str, Any] | Any,
                      config: RunnableConfig | None = None,
                      *,
                      stream_mode: StreamMode = "values",
                      output_keys: str | Sequence[str] | None = None,
                      interrupt_before: All | Sequence[str] | None = None,
                      interrupt_after: All | Sequence[str] | None = None,
                      checkpoint_during: bool | None = None,
                      debug: bool | None = None,
                      **kwargs: Any) -> dict[str, Any] | Any:
        
        return await self._compiled_graph.ainvoke(
            input, 
            config=config, 
            stream_mode=stream_mode, 
            output_keys=output_keys, 
            interrupt_before=interrupt_before, 
            interrupt_after=interrupt_after, 
            checkpoint_during=checkpoint_during, 
            debug=debug, 
            **kwargs
        )

    @requires_compile
    async def astream(self,
                      input: dict[str, Any] | Any,
                      config: RunnableConfig | None = None,
                      *,
                      stream_mode: StreamMode | list[StreamMode] | None = None,
                      output_keys: str | Sequence[str] | None = None,
                      interrupt_before: All | Sequence[str] | None = None,
                      interrupt_after: All | Sequence[str] | None = None,
                      checkpoint_during: bool | None = None,
                      debug: bool | None = None,
                      subgraphs: bool = False,
                      **kwargs: Any) -> AsyncIterator[dict[str, Any] | Any]:
        
        async for chunk in self._compiled_graph.astream(
            input, 
            config=config, 
            stream_mode=stream_mode, 
            output_keys=output_keys, 
            interrupt_before=interrupt_before, 
            interrupt_after=interrupt_after, 
            checkpoint_during=checkpoint_during, 
            debug=debug, 
            subgraphs=subgraphs, 
            **kwargs
        ):
            yield chunk

    @requires_compile
    async def astream_events(self,
                             input: Any,
                             config: Optional[RunnableConfig] = None,
                             *,
                             version: Literal["v1", "v2"] = "v2",
                             include_names: Optional[Sequence[str]] = None,
                             include_types: Optional[Sequence[str]] = None,
                             include_tags: Optional[Sequence[str]] = None,
                             exclude_names: Optional[Sequence[str]] = None,
                             exclude_types: Optional[Sequence[str]] = None,
                             exclude_tags: Optional[Sequence[str]] = None,
                             **kwargs: Any) -> AsyncIterator[StreamEvent]:
        
        async for event in self._compiled_graph.astream_events(
            input, 
            config=config,
            version=version,
            include_names=include_names,
            include_types=include_types,
            include_tags=include_tags,
            exclude_names=exclude_names,
            exclude_types=exclude_types,
            exclude_tags=exclude_tags,
            **kwargs
        ):
            yield event

    @requires_compile
    async def abatch(self,
                     inputs: list[Input],
                     config: Optional[Union[RunnableConfig, list[RunnableConfig]]] = None,
                     *,
                     return_exceptions: bool = False,
                     **kwargs: Optional[Any]) -> list[Output]:

        return await self._compiled_graph.abatch(
            inputs, 
            config=config, 
            return_exceptions=return_exceptions, 
            **kwargs
        )

if __name__ == "__main__":
    async def test_base_agent():
        output_lines = []
        
        try:
            checkpointer = MemorySaver()
            
            base_agent = BaseAgent(
                model_kwargs={},
                vlm_kwargs={},
                node_kwargs={},
                debug=False
            )
            
            compiled_agent = base_agent.compile(
                checkpointer=checkpointer,
                store=None,
                interrupt_before=None,
                interrupt_after=None,
                debug=False,
                name="test_agent"
            )
            
            config = {"configurable": {"thread_id": "test_thread"}}
            
            def capture_checkpoints(round_name):
                output_lines.append(f"\n=== CHECKPOINTS AFTER {round_name} ===")
                try:
                    checkpoints = list(checkpointer.list(config))
                    output_lines.append(f"Total checkpoints: {len(checkpoints)}")
                    
                    for i, checkpoint_tuple in enumerate(checkpoints):
                        output_lines.append(f"\n--- Checkpoint {i+1} ---")
                        output_lines.append(f"Checkpoint ID: {checkpoint_tuple.checkpoint['id']}")
                        output_lines.append(f"Checkpoint TS: {checkpoint_tuple.checkpoint['ts']}")
                        output_lines.append(f"Config: {checkpoint_tuple.config}")
                        
                        if 'channel_values' in checkpoint_tuple.checkpoint:
                            output_lines.append("Channel Values:")
                            for key, value in checkpoint_tuple.checkpoint['channel_values'].items():
                                if key == 'messages':
                                    output_lines.append(f"  {key}: {len(value)} messages")
                                    for j, msg in enumerate(value):
                                        msg_type = type(msg).__name__
                                        content_preview = str(msg.content)[:100] + "..." if len(str(msg.content)) > 100 else str(msg.content)
                                        output_lines.append(f"    Message {j+1} ({msg_type}): {content_preview}")
                                else:
                                    output_lines.append(f"  {key}: {value}")
                except Exception as e:
                    output_lines.append(f"Error capturing checkpoints: {e}")
            
            output_lines.append("=== Testing Immutable BaseAgent ===")
            
            capture_checkpoints("INITIAL")
            
            output_lines.append("\n--- Round 1: Initial image question ---")
            round1_input = {"messages": [
                SystemMessage(content="You are a helpful assistant with vision capabilities."),
                HumanMessage(content=[
                    {"type": "text", "text": "What do you see in this image?"},
                    {"type": "image_url", "image_url": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"}}
                ])
            ]}
            
            result1 = await compiled_agent.ainvoke(round1_input, config=config)
            print(f"Result 1: {result1}")
            output_lines.append(f"Result 1: {result1}")
            
            capture_checkpoints("ROUND 1")
            
            output_lines.append("\n--- Round 2: Follow-up about colors ---")
            round2_input = {"messages": [HumanMessage(content="Can you describe the colors in more detail?")]}
            
            result2 = await compiled_agent.ainvoke(round2_input, config=config)
            print(f"Result 2: {result2}")
            output_lines.append(f"Result 2: {result2}")
            
            capture_checkpoints("ROUND 2")
            
            output_lines.append("\n--- Testing immutable bind_tools ---")
            
            def dummy_tool(query: str) -> str:
                """A simple dummy tool that returns a formatted response for any query."""
                return f"Tool result for: {query}"
            
            agent_with_tools = compiled_agent.bind_tools([dummy_tool])
            
            tools_agent = agent_with_tools.compile(
                checkpointer=checkpointer,
                name="tools_agent"
            )
            
            output_lines.append("Created new agent with tools (immutable)")
            
            round3_input = {"messages": [HumanMessage(content="Use a tool to help me")]}
            
            result3 = await tools_agent.ainvoke(round3_input, config=config)
            output_lines.append(f"Result 3 (with tools): {result3}")
            print(f"Result 3 (with tools): {result3}")
            
            capture_checkpoints("ROUND 3")
            
            output_lines.append("\n--- Testing concurrent access safety ---")
            
            async def concurrent_task(agent, task_id):
                task_config = {"configurable": {"thread_id": f"task_{task_id}"}}
                task_input = {"messages": [HumanMessage(content=[
                    {"type": "text", "text": "What do you see in this image?"},
                    {"type": "image_url", "image_url": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"}}
                ])]}
                try:
                    result = await agent.ainvoke(task_input, config=task_config)
                    return f"Task {task_id} completed: {result}"
                except Exception as e:
                    return f"Task {task_id} failed: {e}"
            
            tasks = [concurrent_task(compiled_agent, i) for i in range(100)]
            concurrent_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in concurrent_results:
                print(f"Concurrent result: {result}")
                output_lines.append(f"Concurrent result: {result}")
            
            output_lines.append("\n--- Testing error handling ---")
            
            try:
                uncompiled_agent = BaseAgent()
                await uncompiled_agent.ainvoke({"messages": [HumanMessage(content="Test")]})
            except ValueError as e:
                output_lines.append(f"Expected error caught: {e}")
            
            output_lines.append("\n=== Test completed successfully ===")
            
        except Exception as e:
            output_lines.append(f"Test failed with error: {e}")
            import traceback
            output_lines.append(f"Traceback: {traceback.format_exc()}")
        
        with open("single_shot_output.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(output_lines))
        
        print(f"\nOutput saved to single_shot_output.txt ({len(output_lines)} lines)")

    asyncio.run(test_base_agent())

