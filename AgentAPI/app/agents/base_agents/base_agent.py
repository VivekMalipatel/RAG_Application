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
from langgraph.cache.base import BaseCache
from langchain_core.tools import BaseTool
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.schema import StreamEvent
from langchain_core.runnables.utils import Output
from langchain_core.runnables.base import Input
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage, ToolMessage
from langgraph.config import get_stream_writer
from openai import BaseModel
from agents.utils import _load_prompt
from pathlib import Path
from llm.llm import LLM
from embed.embed import JinaEmbeddings
from agents.base_agents.base_state import BaseState
from agents.base_agents.memory.base_checkpointer import BaseMemorySaver
from agents.base_agents.memory.base_store import BaseMemoryStore
from agents.base_agents.utils import (
    count_tokens, optimize_messages_for_tokens, get_messages_to_save, 
    generate_message_id, should_trim_state, find_second_last_ai_index
)
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
        if not self._compiled_graph or self._needs_compilation:
            await self.compile()
            self._needs_compilation = False
        return await fn(self, *args, **kwargs)
    return wrapper


def requires_compile_generator(fn):
    @wraps(fn)
    async def wrapper(self, *args, **kwargs):
        if not self._compiled_graph or self._needs_compilation:
            await self.compile()
            self._needs_compilation = False
        async for item in fn(self, *args, **kwargs):
            yield item
    return wrapper


class BaseAgent:

    def __init__(self,
                 prompt: Optional[str] = "You are a helpful assistant.",
                 *,
                 config: RunnableConfig,
                 model_kwargs: Optional[dict[str, Any]] = None,
                 vlm_kwargs: Optional[dict[str, Any]] = None,
                 node_kwargs: Optional[dict[str, Any]] = None,
                 recursion_limit: Optional[int] = 25,
                 debug: bool = False):
        
        if not config:
            raise ValueError("Runnable Config with user_id, thread_id, org_id and/or checkpoint_id must be provided.")

        self._config = AgentConfig(
            model_kwargs=model_kwargs or {},
            vlm_kwargs=vlm_kwargs or {},
            node_kwargs=node_kwargs or {},
            debug=debug
        )
        
        self._compiled_graph: Optional[CompiledStateGraph] = None
        self._llm = LLM(self._config.model_kwargs, self._config.vlm_kwargs)
        self._tools: Sequence[Union[typing.Dict[str, Any], type, Callable, BaseTool]] = []
        self._logger = logging.getLogger(__name__)
        self._needs_compilation = False
        self._memory_tasks: list[asyncio.Task] = []
        
        self._checkpointer = None
        self._store = None
        self._interrupt_before = None
        self._interrupt_after = None
        self._name = None
        self._is_structured_output = False
        self._resursion_limit = recursion_limit
        self.config = config

        if prompt is None:
            prompt = "You are a helpful assistant."
        
        base_prompt = _load_prompt("base_agent", base_dir=Path(__file__).parent)
        self.prompt = f"User ID : {config['configurable']['user_id']}, Org ID : {config['configurable']['org_id']}" + prompt + base_prompt

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
        self._is_structured_output = True
        return self

    def bind_tools(self,
                   tools: Sequence[Union[typing.Dict[str, Any], type, Callable, BaseTool]],
                   *,
                   tool_choice: Optional[Union[str]] = None,
                   **kwargs: Any) -> 'BaseAgent':
        
        self._validate_tools(tools)
        self._tools = tools
        self._llm = self._llm.bind_tools(tools, tool_choice=tool_choice, **kwargs)
        self._needs_compilation = True
        return self

    async def remember(self, state: BaseState, config: RunnableConfig):
        user_id = config["configurable"]["user_id"]
        org_id = config["configurable"]["org_id"]
        messages = state["messages"]
        
        messages_to_save = get_messages_to_save(messages)

        task = asyncio.create_task(self._remember_background(user_id, org_id, messages_to_save, config))
        self._memory_tasks.append(task)

        if should_trim_state(messages):
            second_last_ai_index = find_second_last_ai_index(messages)
            if second_last_ai_index != -1:
                return {"messages": messages[second_last_ai_index:]}
        
        return state

    async def _remember_background(self, user_id: str, org_id: str, messages_to_save: list, config: RunnableConfig):
        try:
            namespace = ("memories", user_id)
            checkpoint_id = config.get("checkpoint_id")
            tasks = []
            
            for i, message in enumerate(messages_to_save):
                message_id = generate_message_id(checkpoint_id, org_id, user_id, i)
                task = asyncio.create_task(get_store().aput(
                    namespace, 
                    message_id, 
                    {"data": message.content}
                ))
                tasks.append(task)
            
            if tasks:
                await asyncio.gather(*tasks)
        
        except Exception as e:
            self._logger.error(f"Background remember task failed: {e}")

    async def wait_for_memory_tasks(self):
        if self._memory_tasks:
            try:
                await asyncio.gather(*self._memory_tasks, return_exceptions=True)
                self._logger.debug(f"Completed {len(self._memory_tasks)} memory tasks")
            except Exception as e:
                self._logger.error(f"Error waiting for memory tasks: {e}")
            finally:
                self._memory_tasks.clear()

    async def retrieve_memory(self, state: BaseState, config: RunnableConfig):
        user_id = config["configurable"]["user_id"]
        org_id = config["configurable"]["org_id"]
        last_message: BaseMessage = state["messages"][-1]
        writer = get_stream_writer()
        writer(f"Retrieving memory.....\n\n")
        store = get_store()

        async def get_user_memory():
            namespace = ("memories", user_id)
            try:
                search_query = last_message.content
                result = await store.asearch(namespace, query=search_query, limit=envconfig.MAX_MEMORY_SEARCH_RESULTS)
                return result
            except Exception as e:
                self._logger.error(f"User memory search failed: {e}")
                return []
        
        async def get_org_memory():
            namespace = ("memories", org_id)
            try:
                search_query = last_message.content
                result = await store.asearch(namespace, query=search_query, limit=envconfig.MAX_MEMORY_SEARCH_RESULTS)
                return result
            except Exception as e:
                self._logger.error(f"Org memory search failed: {e}")
                return []
        
        user_memory_task = asyncio.create_task(get_user_memory())
        org_memory_task = asyncio.create_task(get_org_memory())
        
        user_memory = await user_memory_task
        org_memory = await org_memory_task

        if not user_memory and not org_memory:
            writer(f"Memory retrieved......\n\n")
            return []
        
        memory_messages = []

        async def process_user_memory():
            if not user_memory:
                return [SystemMessage(content="No user memory found.")]
            return [SystemMessage(content=user_msg.value.get("data", "")) for user_msg in user_memory]

        async def process_org_memory():
            if not org_memory:
                return [SystemMessage(content="No organization memory found.")]
            return [SystemMessage(content=org_msg.value.get("data", "")) for org_msg in org_memory]

        user_messages_task = asyncio.create_task(process_user_memory())
        org_messages_task = asyncio.create_task(process_org_memory())
        
        user_messages = await user_messages_task
        org_messages = await org_messages_task
        
        memory_messages.extend(user_messages)
        memory_messages.extend(org_messages)

        writer(f"Memory retrieved......\n\n")
        return memory_messages

    async def llm_node(self, state: BaseState, config: RunnableConfig):
        memory_messages = await self.retrieve_memory(state, config)

        system_messages = [SystemMessage(content=self.prompt)]
        memory_wrapper = [
            HumanMessage(content="<Retrieved Messages from Memory Start>"),
            *memory_messages,
            HumanMessage(content="<Retrieved Messages from Memory End>")
        ]
        state_messages = state["messages"]
        
        optimized_system, optimized_memory, optimized_state, was_optimized, trimmed_messages = optimize_messages_for_tokens(
            system_messages,
            memory_wrapper,
            state_messages,
            envconfig.MAX_STATE_TOKENS,
            self._logger
        )
        
        if trimmed_messages:
            user_id = config["configurable"]["user_id"]
            org_id = config["configurable"]["org_id"]
            self._logger.debug(f"Storing {len(trimmed_messages)} trimmed messages in background")
            task = asyncio.create_task(self._remember_background(user_id, org_id, trimmed_messages, config))
            self._memory_tasks.append(task)
        
        messages = optimized_system + optimized_memory + optimized_state
        
        response = await self._llm.ainvoke(messages, **self._config.node_kwargs)
        
        if self._is_structured_output:
            response = AIMessage(content=json.dumps(response))
        
        if was_optimized and optimized_state != state_messages:
            self._logger.debug(f"Returning optimized state: {len(optimized_state)} messages + new response")
            return {"messages": optimized_state + [response]}
        else:
            return {"messages": [response]}

    def _compile_graph(self, has_tools: bool, **compile_kwargs) -> CompiledStateGraph:
        graph_builder = StateGraph(BaseState)
        llm_node_name = f"llm${self.config.get('configurable').get('user_id')}"
        graph_builder.add_node(llm_node_name, self.llm_node)
        graph_builder.add_node("remember_node", self.remember)
        
        if has_tools:
            graph_builder.add_node("tools", ToolNode(self._tools))
            graph_builder.add_edge(START, llm_node_name)
            graph_builder.add_conditional_edges(
                llm_node_name,
                tools_condition,
                {"tools": "tools", "__end__": "remember_node"}
            )
            graph_builder.add_edge("tools", llm_node_name)
            graph_builder.add_edge("remember_node", END)
        else:
            graph_builder.add_edge(START, llm_node_name)
            graph_builder.add_edge(llm_node_name, "remember_node")
            graph_builder.add_edge("remember_node", END)
        
        compiled_graph = graph_builder.compile(**compile_kwargs)
        
        return compiled_graph

    async def compile(self,
                      checkpointer: Optional[BaseMemorySaver] = None,
                      *,
                      cache: BaseCache | None = None,
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
                "embed": JinaEmbeddings(
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
            "cache": cache,
            "store": store,
            "interrupt_before": interrupt_before,
            "interrupt_after": interrupt_after,
            "debug": debug if debug is not None else self._config.info,
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
                      print_mode: StreamMode | Sequence[StreamMode] = (),
                      output_keys: str | Sequence[str] | None = None,
                      interrupt_before: All | Sequence[str] | None = None,
                      interrupt_after: All | Sequence[str] | None = None,
                      **kwargs: Any) -> dict[str, Any] | Any:
        
        config["recursion_limit"] = self._resursion_limit
        #TODO : Temperorily disable thinking
        if "extra_body" not in config:
            config["extra_body"] = {}
        # if "chat_template_kwargs" not in config["extra_body"]:
        #     config["extra_body"]["chat_template_kwargs"] = {}
        # # config["extra_body"]["chat_template_kwargs"]["enable_thinking"]=True
        config["extra_body"]["top_k"]=envconfig.REASONING_LLM_TOP_K
        config["extra_body"]["min_p"]=envconfig.REASONING_LLM_MIN_P
        config["extra_body"]["repetition_penalty"] = envconfig.REASONING_LLM_REPETITION_PENALTY
        
        result = await self._compiled_graph.ainvoke(
            input, 
            config=config, 
            stream_mode=stream_mode,
            output_keys=output_keys, 
            print_mode=print_mode,
            interrupt_before=interrupt_before, 
            interrupt_after=interrupt_after, 
            **kwargs
        )
        
        return result

    @requires_compile_generator
    async def astream(self,
                      input: dict[str, Any] | Any,
                      config: RunnableConfig | None = None,
                      *,
                      stream_mode: StreamMode | Sequence[StreamMode] | None = None,
                      print_mode: StreamMode | Sequence[StreamMode] = (),
                      output_keys: str | Sequence[str] | None = None,
                      interrupt_before: All | Sequence[str] | None = None,
                      interrupt_after: All | Sequence[str] | None = None,
                      checkpoint_during: bool | None = None,
                      debug: bool | None = None,
                      subgraphs: bool = False,
                      **kwargs: Any) -> AsyncIterator[dict[str, Any] | Any]:
        
        if isinstance(stream_mode, str):
            async for chunk in self._compiled_graph.astream(
                input, 
                config=config, 
                stream_mode=stream_mode, 
                print_mode=print_mode,
                output_keys=output_keys, 
                interrupt_before=interrupt_before, 
                interrupt_after=interrupt_after, 
                checkpoint_during=checkpoint_during, 
                debug=debug, 
                subgraphs=subgraphs, 
                **kwargs
            ):
                yield chunk
        elif isinstance(stream_mode, list):
            async for stream_mode, chunk in self._compiled_graph.astream(
                input, 
                config=config, 
                stream_mode=stream_mode, 
                print_mode=print_mode,
                output_keys=output_keys, 
                interrupt_before=interrupt_before, 
                interrupt_after=interrupt_after, 
                checkpoint_during=checkpoint_during, 
                debug=debug, 
                subgraphs=subgraphs, 
                **kwargs
            ):
                yield stream_mode, chunk

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

    @property
    def graph(self) -> Optional[CompiledStateGraph]:
        """Access the compiled graph. Returns None if not compiled."""
        return self._compiled_graph


if __name__ == "__main__":
    async def test_base_agent():
        output_lines = []

        config = {
            "configurable": {
                "thread_id": "test_thread1", 
                "user_id": "test_user1", 
                "org_id": "test_org1"
            }
        }

        base_agent = BaseAgent(
            model_kwargs={},
            vlm_kwargs={},
            node_kwargs={},
            debug=False,
            config=config
        )

        compiled_agent: BaseAgent = await base_agent.compile(name="test_agent")

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
    
    
    