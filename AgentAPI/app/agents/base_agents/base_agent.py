import uuid
import typing
import asyncio
import logging
import json
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Sequence, Union, Optional, Callable, AsyncIterator, Literal

from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.store.base import BaseStore, IndexConfig
from langgraph.types import All, StreamMode
from langgraph.config import get_store

from langchain_core.tools import BaseTool, tool
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.schema import StreamEvent
from langchain_core.runnables.utils import Output
from langchain_core.runnables.base import Input
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_openai import OpenAIEmbeddings

from openai import BaseModel

from llm.llm import LLM
from agents.base_agents.base_state import BaseState
from agents.base_agents.memory.base_checkpointer import BaseMemorySaver
from agents.base_agents.memory.base_store import BaseMemoryStore
from db.redis import redis
from config import config as envconfig

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


def requires_compile_generator(fn):
    @wraps(fn)
    async def wrapper(self, *args, **kwargs):
        if not self._compiled_graph:
            raise ValueError("Agent not compiled. Call compile() first.")
        async for item in fn(self, *args, **kwargs):
            yield item
    return wrapper


class BaseAgent:

    def __init__(self,
                 prompt: Optional[str] = "You are a helpful assistant.",
                 *,
                 model_kwargs: Optional[dict[str, Any]] = None,
                 vlm_kwargs: Optional[dict[str, Any]] = None,
                 node_kwargs: Optional[dict[str, Any]] = None,
                 recursion_limit: Optional[int] = 25,
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
        self._is_structred_output= False
        self._resursion_limit = recursion_limit
        
        self.prompt = prompt

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

    def with_structured_output(self, schema: Union[dict, type[BaseModel]], **kwargs: Any) -> 'BaseAgent':
        self._llm = self._llm.with_structured_output(schema=schema, **kwargs)
        self._is_structred_output= True
        return self

    def bind_tools(self,
                   tools: Sequence[Union[typing.Dict[str, Any], type, Callable, BaseTool]],
                   *,
                   tool_choice: Optional[Union[str]] = None,
                   **kwargs: Any) -> 'BaseAgent':
        
        self._validate_tools(tools)
        self._tools = tools
        self._llm = self._llm.bind_tools(tools, tool_choice=tool_choice, **kwargs)
        self.compile()
        return self

    async def remember(self, state: BaseState, config: RunnableConfig):
        user_id = config["configurable"]["user_id"]
        org_id = config["configurable"]["org_id"]
        namespace = ("memories", user_id)
        messages = state["messages"]
        second_last_ai_index = -1
        ai_count = 0
        
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], AIMessage):
                ai_count += 1
                if ai_count == 2:
                    second_last_ai_index = i
                    break
        
        if second_last_ai_index != -1:
            messages_to_save = messages[second_last_ai_index + 1:]
        else:
            messages_to_save = messages
        
        checkpoint_id = config.get("checkpoint_id")
        for i, message in enumerate(messages_to_save):
            asyncio.create_task(get_store().aput(
                namespace, 
                f"{checkpoint_id}_{i}" if checkpoint_id else f"{org_id}_{user_id}_{str(uuid.uuid4())}_{i}", 
                {"data": message.content}
            ))

        num_tokens = state["messages"][-1].usage_metadata.get("total_tokens", 0) if state["messages"][-1].usage_metadata else 0
        self._logger.info(f"State tokens after saving: {num_tokens}")

        if num_tokens >= envconfig.MAX_STATE_TOKENS:
            if second_last_ai_index != -1:
                state["messages"] = messages[second_last_ai_index:]
            return state
        
        return

    async def retrieve_memory(self, state: BaseState, config: RunnableConfig):
        user_id = config["configurable"]["user_id"]
        org_id = config["configurable"]["org_id"]
        last_message: BaseMessage = state["messages"][-1]
        store = get_store()
        
        async def get_user_memory():
            namespace = ("memories", user_id)
            try:
                return await store.asearch(namespace, query=str(last_message.content))
            except Exception as e:
                self._logger.error(f"Error retrieving user memory: {e}")
                return []
        
        async def get_org_memory():
            namespace = ("memories", org_id)
            try:
                return await store.asearch(namespace, query=str(last_message.content))
            except Exception as e:
                self._logger.error(f"Error retrieving organization memory: {e}")
                return []
        
        user_memory, org_memory = await asyncio.gather(get_user_memory(), get_org_memory())

        if not user_memory and not org_memory:
            return []
        
        memory_messages = []

        if not user_memory:
            memory_messages.append(SystemMessage(content="No user memory found."))
        else:
            for user_msg in user_memory:
                memory_messages.append(SystemMessage(content=str(user_msg.value.get("data", ""))))

        if not org_memory:
            memory_messages.append(SystemMessage(content="No organization memory found."))
        else:
            for org_msg in org_memory:
                memory_messages.append(SystemMessage(content=str(org_msg.value.get("data", ""))))

        return memory_messages

    async def llm_node(self, state: BaseState, config: RunnableConfig):
        memory_messages = await self.retrieve_memory(state, config)
        messages = [SystemMessage(content=self.prompt)] + memory_messages + state["messages"]
        
        response = await self._llm.ainvoke(messages, **self._config.node_kwargs)
        if self.is_structred_output:
            response = AIMessage(content=json.dumps(response))
        return {
            "messages": [response], 
            "user_id": config["configurable"]["user_id"], 
            "org_id": config["configurable"]["org_id"]
        }


    def _compile_graph(self, has_tools: bool, **compile_kwargs) -> CompiledStateGraph:
        graph_builder = StateGraph(BaseState)
        
        graph_builder.add_node("llm", self.llm_node)
        graph_builder.add_node("remember_node", self.remember)
        
        if has_tools:
            graph_builder.add_node("tools", ToolNode(self._tools))
            graph_builder.add_edge(START, "llm")
            graph_builder.add_conditional_edges(
                "llm",
                tools_condition,
                {"tools": "tools", "__end__": "remember_node"}
            )
            graph_builder.add_edge("tools", "llm")
            graph_builder.add_edge("remember_node", END)
        else:
            graph_builder.add_edge(START, "llm")
            graph_builder.add_edge("llm", "remember_node")
            graph_builder.add_edge("remember_node", END)

        return graph_builder.compile(**compile_kwargs)

    async def compile(self,
                      checkpointer: Optional[BaseMemorySaver] = None,
                      *,
                      store: Optional[BaseStore] = None,
                      interrupt_before: list[str] | Literal['*'] | None = None,
                      interrupt_after: list[str] | Literal['*'] | None = None,
                      debug: bool = False,
                      name: str | None = None) -> 'BaseAgent':

        self._checkpointer = checkpointer
        if checkpointer is None:
            checkpointer = BaseMemorySaver(redis_client=redis.get_session())
            self._checkpointer = checkpointer
            await self._checkpointer.asetup()

        self._store = store
        if store is None:
            index_config: IndexConfig = {
                "dims": envconfig.MULTIMODEL_EMBEDDING_MODEL_DIMS,
                "embed": OpenAIEmbeddings(
                    model=envconfig.MULTIMODEL_EMBEDDING_MODEL,
                    base_url=envconfig.OPENAI_BASE_URL, 
                    api_key=envconfig.OPENAI_API_KEY
                ),
                "ann_index_config": {"vector_type": "vector"},
                "distance_type": "cosine",
            }

            store = BaseMemoryStore(
                redis_client=redis.get_session(),
                index=index_config,
            )
            self._store = store
            await self._store.setup()

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
        
        config["recursion_limit"] = self._resursion_limit
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

    @requires_compile_generator
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

        base_agent = BaseAgent(
            model_kwargs={},
            vlm_kwargs={},
            node_kwargs={},
            debug=False
        )

        compiled_agent: BaseAgent = await base_agent.compile(name="test_agent")
        
        config = {
            "configurable": {
                "thread_id": "test_thread", 
                "user_id": "test_user", 
                "org_id": "test_org"
            }
        }
        
        async def capture_checkpoints(round_name):
            output_lines.append(f"\n=== CHECKPOINTS AFTER {round_name} ===")
            try:
                checkpoints = [checkpoint async for checkpoint in compiled_agent._checkpointer.alist(config)]
                output_lines.append(f"Total checkpoints: {len(checkpoints)}")
                
                for i, checkpoint_tuple in enumerate(checkpoints):
                    output_lines.append(f"\n--- Checkpoint {i+1} ---")
                    output_lines.append(str(checkpoint_tuple))
            except Exception as e:
                output_lines.append(f"Error capturing checkpoints: {e}")
        
        output_lines.append("=== Testing Immutable BaseAgent ===")
        
        await capture_checkpoints("INITIAL")
        
        output_lines.append("\n--- Round 1: Initial image question ---")
        round1_input = {
            "messages": [
                HumanMessage(content=[
                    {"type": "text", "text": "What do you see in this image?"},
                    {"type": "image_url", "image_url": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"}}
                ])
            ],
            "user_id": config["configurable"]["user_id"],
            "org_id": config["configurable"]["org_id"]
        }
        
        result1 = await compiled_agent.ainvoke(round1_input, config=config)
        print(f"Result 1: {result1}")
        output_lines.append(f"Result 1: {result1}")
        
        await capture_checkpoints("ROUND 1")
        
        output_lines.append("\n--- Round 2: Follow-up about colors ---")
        round2_input = {"messages": [HumanMessage(content="Can you describe the colors in more detail?")]}
        
        result2 = await compiled_agent.ainvoke(round2_input, config=config)
        print(f"Result 2: {result2}")
        output_lines.append(f"Result 2: {result2}")
        
        with open("single_shot_output.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(output_lines))
        
        print(f"\nOutput saved to single_shot_output.txt ({len(output_lines)} lines)")

    asyncio.run(test_base_agent())