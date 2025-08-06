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
from langgraph.cache.base import BaseCache
from langchain_core.tools import BaseTool
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.schema import StreamEvent
from langchain_core.runnables.utils import Output
from langchain_core.runnables.base import Input
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langgraph.config import get_stream_writer
from openai import BaseModel
from agents.utils import _load_prompt
from pathlib import Path
from llm.llm import LLM
from embed.embed import JinaEmbeddings
from agents.base_agents.base_state import BaseState
from agents.base_agents.memory.base_checkpointer import BaseMemorySaver
from agents.base_agents.memory.base_store import BaseMemoryStore
from db.redis import redis
from config import config as envconfig
from core.background_executor import submit_background_task_with_redis

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
        self.prompt = prompt + base_prompt

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
        self.compile()
        return self

    async def remember(self, state: BaseState, config: RunnableConfig):
        user_id = config["configurable"]["user_id"]
        org_id = config["configurable"]["org_id"]
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

        submit_background_task_with_redis(
            self._remember_background_safe,
            user_id, org_id, messages_to_save, config
        )

        num_tokens = state["messages"][-1].usage_metadata.get("total_tokens", 0) if state["messages"][-1].usage_metadata else 0

        if num_tokens >= envconfig.MAX_STATE_TOKENS:
            if second_last_ai_index != -1:
                return {"messages": messages[second_last_ai_index:]}
        
        return state

    async def _remember_background_safe(self, redis_client, user_id: str, org_id: str, messages_to_save: list, config: RunnableConfig):
        try:
            from agents.base_agents.memory.base_store import BaseMemoryStore
            from embed.embed import JinaEmbeddings
            from config import config as envconfig
            
            logger = logging.getLogger(__name__)
            logger.debug(f"Starting background memory task for user {user_id}, {len(messages_to_save)} messages")
            
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
                redis_client=redis_client,
                index=index_config,
            )
            await store.setup()
            logger.debug("Memory store setup completed")

            namespace = ("memories", user_id)
            checkpoint_id = config.get("checkpoint_id")
            
            for i, message in enumerate(messages_to_save):
                try:
                    message_id = f"{checkpoint_id}_{i}" if checkpoint_id else f"{org_id}_{user_id}_{str(uuid.uuid4())}_{i}"
                    logger.debug(f"Storing message {i+1}/{len(messages_to_save)} with ID: {message_id}")
                    
                    await store.aput(
                        namespace, 
                        message_id, 
                        {"data": message.content}
                    )
                    logger.debug(f"Successfully stored message {i+1}/{len(messages_to_save)}")
                except Exception as e:
                    logger.error(f"Failed to store message {i+1}/{len(messages_to_save)}: {e}")
                    continue
            
            logger.debug(f"Background memory task completed for user {user_id}")
        
        except Exception as e:
            logger.error(f"Background remember task failed: {e}", exc_info=True)

    async def _remember_background(self, user_id: str, org_id: str, messages_to_save: list, config: RunnableConfig):
        try:
            namespace = ("memories", user_id)
            checkpoint_id = config.get("checkpoint_id")
            tasks = []
            
            for i, message in enumerate(messages_to_save):
                task = asyncio.create_task(get_store().aput(
                    namespace, 
                    f"{checkpoint_id}_{i}" if checkpoint_id else f"{org_id}_{user_id}_{str(uuid.uuid4())}_{i}", 
                    {"data": message.content}
                ))
                tasks.append(task)
            
            if tasks:
                await asyncio.gather(*tasks)
        
        except Exception as e:
            self._logger.error(f"Background remember task failed: {e}")

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
                result = await store.asearch(namespace, query=search_query)
                return result
            except Exception as e:
                self._logger.error(f"User memory search failed: {e}")
                return []
        
        async def get_org_memory():
            namespace = ("memories", org_id)
            try:
                search_query = last_message.content
                result = await store.asearch(namespace, query=search_query)
                return result
            except Exception as e:
                self._logger.error(f"Org memory search failed: {e}")
                return []
        
        user_memory, org_memory = await asyncio.gather(get_user_memory(), get_org_memory())

        if not user_memory and not org_memory:
            writer(f"Memory retrieved......\n\n")
            return []
        
        memory_messages = []

        async def process_user_memory():
            if not user_memory:
                return [HumanMessage(content="No user memory found.")]
            return [HumanMessage(content=user_msg.value.get("data", "")) for user_msg in user_memory]

        async def process_org_memory():
            if not org_memory:
                return [HumanMessage(content="No organization memory found.")]
            return [HumanMessage(content=org_msg.value.get("data", "")) for org_msg in org_memory]

        user_messages, org_messages = await asyncio.gather(
            process_user_memory(),
            process_org_memory()
        )
        
        memory_messages.extend(user_messages)
        memory_messages.extend(org_messages)

        writer(f"Memory retrieved......\n\n")
        return memory_messages

    async def llm_node(self, state: BaseState, config: RunnableConfig):
        memory_messages = await self.retrieve_memory(state, config)

        messages = [SystemMessage(content=self.prompt)] + [HumanMessage(content="<Retrieved Messages from Memory Start>")] + memory_messages + [HumanMessage(content="<Retrieved Messages from Memory End>")] + state["messages"]
        
        response = await self._llm.ainvoke(messages, **self._config.node_kwargs)
        
        if self._is_structured_output:
            response = AIMessage(content=json.dumps(response))
        return {
            "messages": [response]
        }

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
        config["extra_body"]={"chat_template_kwargs": {"enable_thinking": False}}
        
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
    
    
    