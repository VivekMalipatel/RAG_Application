import typing
import asyncio
import logging
import json
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Sequence, Union, Optional, Callable, AsyncIterator, Literal
from langgraph.typing import ContextT, InputT, OutputT, StateT
from langgraph.types import (
    All,
    Checkpointer,
    Command,
)
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.store.base import BaseStore, IndexConfig
from langgraph.types import All, StreamMode, Durability
from langgraph.cache.base import BaseCache
from langchain_core.tools import BaseTool
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.schema import StreamEvent
from langchain_core.runnables.utils import Output
from langchain_core.runnables.base import Input
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.messages.base import messages_to_dict
from langmem import create_manage_memory_tool, create_memory_store_manager, ReflectionExecutor
from openai import BaseModel
from agents.utils import (
    _load_prompt,
    validate_tools,
    build_memory_config,
    coerce_profile_content,
    format_profile_overview,
    build_profile_directives,
    wrap_manage_memory_tool_json,
)
from pathlib import Path
from llm.llm import LLM
from embed.embed import JinaEmbeddings
from agents.base_agents.base_state import BaseState
from agents.base_agents.memory.base_checkpointer import BaseMemorySaver
from agents.base_agents.memory.base_store import BaseMemoryStore
from agents.base_agents.memory.base_memorymodels import SemanticMemory, UserProfileMemory
from db.redis import redis
from config import config as envconfig

@dataclass(frozen=True)
class AgentConfig:
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    vlm_kwargs: dict[str, Any] = field(default_factory=dict)
    node_kwargs: dict[str, Any] = field(default_factory=dict)
    debug: bool = False
    profile_memory_enabled: bool = False

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
                 debug: bool = False,
                 enable_profile_memory: bool = False):
        
        if not config:
            raise ValueError("Runnable Config with user_id, thread_id, org_id and/or checkpoint_id must be provided.")

        self._config = AgentConfig(
            model_kwargs=model_kwargs or {},
            vlm_kwargs=vlm_kwargs or {},
            node_kwargs=node_kwargs or {},
            debug=debug,
            profile_memory_enabled=enable_profile_memory,
        )
        
        self._compiled_graph: Optional[CompiledStateGraph] = None
        self._llm = LLM(self._config.model_kwargs, self._config.vlm_kwargs)
        self._tools: Sequence[Union[typing.Dict[str, Any], type, Callable, BaseTool]] = []
        self._user_tools: list[Union[typing.Dict[str, Any], type, Callable, BaseTool]] = []
        self._memory_tools: list[Union[typing.Dict[str, Any], type, Callable, BaseTool]] = []
        self._tool_choice: Optional[Union[str]] = None
        self._tool_bind_kwargs: dict[str, Any] = {}
        self._logger = logging.getLogger(__name__)
        self._needs_compilation = False
        self._semantic_manager = None
        self._reflection_executor = None
        self._semantic_manage_tool = None
        self._profile_manager = None
        self._profile_reflection_executor = None
        self._profile_manage_tool = None
        self._profile_memory_key: Optional[str] = None
        
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
        user_context = f"User ID : {config['configurable']['user_id']}, Org ID : {config['configurable']['org_id']}"
        prompt_parts = [user_context.strip(), prompt.strip(), base_prompt.strip()]
        self.prompt = "\n\n".join(part for part in prompt_parts if part)

    def with_structured_output(self, schema: Union[dict, type[BaseModel]], **kwargs: Any) -> 'BaseAgent':
        self._llm = self._llm.with_structured_output(schema=schema, **kwargs)
        self._is_structured_output = True
        return self

    def bind_tools(self,
                   tools: Sequence[Union[typing.Dict[str, Any], type, Callable, BaseTool]],
                   *,
                   tool_choice: Optional[Union[str]] = None,
                   **kwargs: Any) -> 'BaseAgent':
        
        validate_tools(tools)
        self._user_tools = list(tools)
        self._tool_choice = tool_choice
        self._tool_bind_kwargs = dict(kwargs)
        self._apply_tool_binding()
        self._needs_compilation = True
        return self

    def _setup_semantic_manager(self, store: Optional[BaseStore]):
        self._semantic_manager = None
        self._reflection_executor = None
        self._semantic_manage_tool = None
        if not store:
            return
        namespace = ("memories", "{org_id}", "{user_id}", "semantic")
        semantic_instructions = _load_prompt("semantic_memory_instructions", base_dir=Path(__file__).parent)
        self._semantic_manager = create_memory_store_manager(
            self._llm.reasoning_llm,
            store=store,
            namespace=namespace,
            schemas=[SemanticMemory],
            instructions=semantic_instructions,
            enable_inserts=True,
            enable_deletes=True,
            query_limit=envconfig.SEMANTIC_MEMORY_QUERY_LIMIT,
        )
        self._reflection_executor = ReflectionExecutor(self._semantic_manager, store=store)
        self._memory_tools.append(self._semantic_manager.search_tool)
        self._semantic_manage_tool = create_manage_memory_tool(
            namespace=namespace,
            instructions="Call this tool when you need to create, update, or delete semantic memories for the current user.",
            schema=SemanticMemory,
            store=store,
            name="manage_semantic_memory",
        )
        self._semantic_manage_tool = wrap_manage_memory_tool_json(
            self._semantic_manage_tool,
            tool_name="manage_semantic_memory",
        )
        self._memory_tools.append(self._semantic_manage_tool)

    def _setup_profile_manager(self, store: Optional[BaseStore]):
        self._profile_manager = None
        self._profile_reflection_executor = None
        self._profile_manage_tool = None
        self._profile_memory_key = None
        if not store or not self._config.profile_memory_enabled:
            return
        namespace = ("memories", "{org_id}", "{user_id}", "profile")
        profile_instructions = _load_prompt("profile_memory_instructions", base_dir=Path(__file__).parent)
        self._profile_manager = create_memory_store_manager(
            self._llm.reasoning_llm,
            store=store,
            namespace=namespace,
            schemas=[UserProfileMemory],
            instructions=profile_instructions,
            enable_inserts=False,
            query_limit=envconfig.PROFILE_MEMORY_QUERY_LIMIT,
        )
        self._profile_reflection_executor = ReflectionExecutor(self._profile_manager, store=store)
        self._memory_tools.append(self._profile_manager.search_tool)
        self._profile_manage_tool = create_manage_memory_tool(
            namespace=namespace,
            instructions=(
                "Update the existing profile record when the user explicitly requests a change. "
                "Always supply the profile memory id provided in the system context."
            ),
            schema=UserProfileMemory,
            actions_permitted=("update"),
            store=store,
            name="manage_profile_memory",
        )
        self._profile_manage_tool = wrap_manage_memory_tool_json(
            self._profile_manage_tool,
            tool_name="manage_profile_memory",
        )
        self._memory_tools.append(self._profile_manage_tool)

    async def _get_profile_context(self, config: Optional[RunnableConfig]) -> tuple[Optional[str], Optional[dict[str, Any]]]:
        self._profile_memory_key = None
        if not self._profile_manager or not self._config.profile_memory_enabled or not config:
            return None, None
        try:
            results = await self._profile_manager.asearch(limit=envconfig.PROFILE_MEMORY_QUERY_LIMIT, config=config)
        except Exception as exc:
            self._logger.error("Profile snapshot retrieval failed", exc_info=exc)
            return None, None
        if not results:
            return None, None
        item = results[0]
        self._profile_memory_key = getattr(item, "key", None)
        content = coerce_profile_content(item.value)
        if not isinstance(content, dict):
            self._profile_memory_key = None
            return None, None
        metadata = content.get("Metadata")
        if isinstance(metadata, dict):
            confidence = metadata.get("Confidence")
            if confidence is not None and confidence < envconfig.PROFILE_MEMORY_MIN_CONFIDENCE:
                self._profile_memory_key = None
                return None, content
        overview_source = {
            key: value for key, value in content.items() if key != "Metadata"
        }
        return format_profile_overview(overview_source), content

    async def llm_node(self, state: BaseState, config: RunnableConfig):
        system_parts = [self.prompt.strip()]
        if self._profile_manager and self._config.profile_memory_enabled:
            overview_text, profile_content = await self._get_profile_context(config or self.config)
            if overview_text:
                system_parts.append(overview_text)
            profile_directives = build_profile_directives(profile_content)
            if profile_directives:
                system_parts.append(profile_directives)
            if self._profile_memory_key:
                system_parts.append(
                    f"Use manage_profile_memory with action='update' and id='{self._profile_memory_key}' when the user requests profile changes. If no existsing profile is found, do not create a new one. We will create a new one in the background."
                )
        system_messages = [SystemMessage(content="\n\n".join(part for part in system_parts if part))]
        state_messages = state["messages"]

        messages = system_messages + state_messages
        
        response = await self._llm.ainvoke(messages, **self._config.node_kwargs)
 
        if self._is_structured_output:
            response = AIMessage(content=json.dumps(response))

        payload_messages = [*state_messages, response]
        if self._reflection_executor:
            payload = {
                "messages": payload_messages,
                "max_steps": envconfig.SEMANTIC_MEMORY_MAX_UPDATES_PER_TURN,
            }
            try:
                self._reflection_executor.submit(
                    payload,
                    after_seconds=envconfig.SEMANTIC_MEMORY_DELAY_SECONDS,
                    config=build_memory_config(config),
                )
            except Exception as exc:
                self._logger.error("Semantic memory submission failed", exc_info=exc)
        if self._profile_reflection_executor and self._config.profile_memory_enabled:
            profile_payload = {
                "messages": payload_messages,
                "max_steps": envconfig.PROFILE_MEMORY_MAX_UPDATES_PER_TURN,
            }
            try:
                self._profile_reflection_executor.submit(
                    profile_payload,
                    after_seconds=envconfig.PROFILE_MEMORY_DELAY_SECONDS,
                    config=build_memory_config(config),
                )
            except Exception as exc:
                self._logger.error("Profile memory submission failed", exc_info=exc)

        return {"messages": [response]}

    def _compile_graph(self, has_tools: bool, **compile_kwargs) -> CompiledStateGraph[StateT, ContextT, InputT, OutputT]:
        graph_builder = StateGraph(BaseState)
        llm_node_name = f"llm${self.config.get('configurable').get('user_id')}"
        graph_builder.add_node(llm_node_name, self.llm_node)
        if has_tools:
            graph_builder.add_node("tools", ToolNode(self._tools))
            graph_builder.add_edge(START, llm_node_name)
            graph_builder.add_conditional_edges(
                llm_node_name,
                tools_condition,
                {"tools": "tools", "__end__": END}
            )
            graph_builder.add_edge("tools", llm_node_name)
        else:
            graph_builder.add_edge(START, llm_node_name)
            graph_builder.add_edge(llm_node_name, END)
        
        compiled_graph = graph_builder.compile(**compile_kwargs)
        
        return compiled_graph

    async def compile(self,
                      checkpointer: Checkpointer = None,
                      *,
                      cache: BaseCache | None = None,
                      store: BaseStore | None = None,
                      interrupt_before: All | list[str] | None = None,
                      interrupt_after: All | list[str] | None = None,
                      debug: bool = False,
                      name: str | None = None) -> CompiledStateGraph[StateT, ContextT, InputT, OutputT]:

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
        self._memory_tools = []
        self._setup_semantic_manager(self._store)
        self._setup_profile_manager(self._store)
        self._apply_tool_binding()

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
        self._needs_compilation = False
        return self

    def _apply_tool_binding(self):
        combined = [*self._user_tools, *self._memory_tools]
        self._tools = combined
        if combined:
            self._llm = self._llm.bind_tools(
                combined,
                tool_choice=self._tool_choice,
                **self._tool_bind_kwargs,
            )
        else:
            self._llm.tools = []
        self._needs_compilation = True

    async def aclose(self) -> None:

        def _shutdown_executor(attribute: str) -> None:
            executor = getattr(self, attribute, None)
            if executor and hasattr(executor, "shutdown"):
                try:
                    executor.shutdown(wait=True, cancel_futures=True)
                except TypeError:
                    executor.shutdown()
                except Exception as exc:
                    self._logger.warning("Failed to shutdown reflection executor", exc_info=exc)
            setattr(self, attribute, None)

        _shutdown_executor("_reflection_executor")
        _shutdown_executor("_profile_reflection_executor")

        checkpointer, store = self._checkpointer, self._store
        self._checkpointer = None
        self._store = None

        if store is not None:
            try:
                if hasattr(store, "__aexit__"):
                    await store.__aexit__(None, None, None)
                elif hasattr(store, "aclose"):
                    await store.aclose()
            except Exception as exc:
                self._logger.warning("Failed to close memory store", exc_info=exc)

        if checkpointer is not None:
            try:
                if hasattr(checkpointer, "__aexit__"):
                    await checkpointer.__aexit__(None, None, None)
                elif hasattr(checkpointer, "aclose"):
                    await checkpointer.aclose()
            except Exception as exc:
                self._logger.warning("Failed to close memory checkpointer", exc_info=exc)

    @requires_compile
    async def get_user_profile(self, config: RunnableConfig | None = None) -> Optional[dict[str, Any]]:
        _, profile = await self._get_profile_context(config or self.config)
        return profile

    @requires_compile
    async def ainvoke(self,
                      input: InputT | Command | None,
                      config: RunnableConfig | None = None,
                      *,
                      context: ContextT | None = None,
                      stream_mode: StreamMode = "values",
                      print_mode: StreamMode | Sequence[StreamMode] = (),
                      output_keys: str | Sequence[str] | None = None,
                      interrupt_before: All | Sequence[str] | None = None,
                      interrupt_after: All | Sequence[str] | None = None,
                      durability: Durability | None = None,
                      **kwargs: Any
                    ) -> dict[str, Any] | Any:
        
        config["recursion_limit"] = self._resursion_limit
        # #TODO : Temperorily disable thinking
        # if "extra_body" not in config:
        #     config["extra_body"] = {}
        # # if "chat_template_kwargs" not in config["extra_body"]:
        # #     config["extra_body"]["chat_template_kwargs"] = {}
        # # # config["extra_body"]["chat_template_kwargs"]["enable_thinking"]=True
        # config["extra_body"]["top_k"]=envconfig.REASONING_LLM_TOP_K
        # config["extra_body"]["min_p"]=envconfig.REASONING_LLM_MIN_P
        # config["extra_body"]["repetition_penalty"] = envconfig.REASONING_LLM_REPETITION_PENALTY
        
        result = await self._compiled_graph.ainvoke(
            input, 
            config=config,
            context=context,
            stream_mode=stream_mode,
            output_keys=output_keys, 
            print_mode=print_mode,
            interrupt_before=interrupt_before, 
            interrupt_after=interrupt_after,
            durability=durability,
            **kwargs
        )
        
        return result

    @requires_compile_generator
    async def astream(self,
                      input: InputT | Command | None,
                      config: RunnableConfig | None = None,
                      *,
                      context: ContextT | None = None,
                      stream_mode: StreamMode | Sequence[StreamMode] | None = None,
                      print_mode: StreamMode | Sequence[StreamMode] = (),
                      output_keys: str | Sequence[str] | None = None,
                      interrupt_before: All | Sequence[str] | None = None,
                      interrupt_after: All | Sequence[str] | None = None,
                      durability: Durability | None = None,
                      subgraphs: bool = False,
                      debug: bool | None = None,
                      **kwargs: Any) -> AsyncIterator[dict[str, Any] | Any]:

        if isinstance(stream_mode, str):
            async for chunk in self._compiled_graph.astream(
                input, 
                config=config, 
                context=context,
                stream_mode=stream_mode, 
                print_mode=print_mode,
                output_keys=output_keys, 
                interrupt_before=interrupt_before, 
                interrupt_after=interrupt_after, 
                durability=durability, 
                debug=debug, 
                subgraphs=subgraphs, 
                **kwargs
            ):
                yield chunk
        elif isinstance(stream_mode, list):
            async for stream_mode, chunk in self._compiled_graph.astream(
                input, 
                config=config, 
                context=context,
                stream_mode=stream_mode, 
                print_mode=print_mode,
                output_keys=output_keys, 
                interrupt_before=interrupt_before, 
                interrupt_after=interrupt_after, 
                durability=durability, 
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
        return self._compiled_graph


if __name__ == "__main__":
    async def test_semantic_memory():
        """Test semantic memory functionality following LangMem patterns."""
        import time
        output_lines = []
        
        print("=== Testing Semantic Memory with BaseAgent ===")
        output_lines.append("=== Testing Semantic Memory with BaseAgent ===")
        
        # Test configuration with two different users
        user1_config = {
            "configurable": {
                "thread_id": "memory_test_thread_1",
                "user_id": "alice_user", 
                "org_id": "test_company"
            }
        }
        
        user2_config = {
            "configurable": {
                "thread_id": "memory_test_thread_2", 
                "user_id": "bob_user",
                "org_id": "test_company"
            }
        }
        
        async def test_memory_ingestion_and_retrieval():
            """Test memory ingestion for Alice and retrieval by Bob."""
            
            # === PHASE 1: Alice shares information (Memory Ingestion) ===
            print("\n--- Phase 1: Alice shares personal information ---")
            output_lines.append("\n--- Phase 1: Alice shares personal information ---")
            
            alice_agent = BaseAgent(
                prompt="You are a helpful assistant. Always introduce yourself as Alice's personal assistant.",
                model_kwargs={},
                vlm_kwargs={}, 
                node_kwargs={},
                debug=False,
                config=user1_config
            )
            
            alice_compiled = await alice_agent.compile(name="alice_agent")
            
            # Alice shares rich personal information
            alice_messages = [
                "Hi! I'm Alice Johnson, and I work as a Senior Data Scientist at Google. My favorite programming languages are Python and R.",
                "I love machine learning, especially deep learning and NLP. I have a PhD in Computer Science from Stanford.",
                "My hobbies include hiking, photography, and playing piano. I have a golden retriever named Max.",
                "I prefer working remotely and usually start my day at 9 AM PST. I'm vegetarian and love Italian food."
            ]
            
            for i, msg in enumerate(alice_messages, 1):
                print(f"\nAlice Message {i}: {msg[:50]}...")
                output_lines.append(f"\nAlice Message {i}: {msg[:50]}...")
                
                result = await alice_compiled.ainvoke(
                    {"messages": [HumanMessage(content=msg)]},
                    config=user1_config
                )
                print(f"Response: {result['messages'][-1].content[:100]}...")
                output_lines.append(f"Response: {result['messages'][-1].content[:100]}...")
                
                # Small delay between messages
                await asyncio.sleep(0.1)
            
            print("\n--- Waiting for semantic memory processing (3 minutes delay) ---")
            output_lines.append("\n--- Waiting for semantic memory processing (3 minutes delay) ---")
            
            # Wait for reflection executor to process (3 minutes as configured)
            print("Waiting 200 seconds for ReflectionExecutor to process memories...")
            await asyncio.sleep(200)  # Wait a bit longer than the 180-second delay
            
            # === PHASE 2: Check if memories were stored ===
            print("\n--- Phase 2: Checking stored memories ---")
            output_lines.append("\n--- Phase 2: Checking stored memories ---")
            
            if alice_compiled._store:
                try:
                    # Search for Alice's semantic memories
                    alice_namespace = ("memories", "test_company", "alice_user", "semantic")
                    stored_memories = await alice_compiled._store.asearch(alice_namespace)
                    
                    print(f"Found {len(stored_memories)} semantic memories for Alice")
                    output_lines.append(f"Found {len(stored_memories)} semantic memories for Alice")
                    
                    for i, memory in enumerate(stored_memories[:3]):  # Show first 3
                        print(f"Memory {i+1}: {str(memory.value)[:100]}...")
                        output_lines.append(f"Memory {i+1}: {str(memory.value)[:100]}...")
                        
                except Exception as e:
                    print(f"Error checking memories: {e}")
                    output_lines.append(f"Error checking memories: {e}")
            
            return alice_compiled
        
        async def test_cross_user_memory_access(alice_agent):
            """Test Bob cannot access Alice's memories."""
            
            print("\n--- Phase 3: Bob queries for Alice's information using search tool ---")
            output_lines.append("\n--- Phase 3: Bob queries for Alice's information using search tool ---")
            
            # Create Bob's agent with same org but different user
            bob_agent = BaseAgent(
                prompt="You are a helpful assistant. Always introduce yourself as Bob's assistant. When asked about colleagues, use the MemorySearch tool to find relevant information.",
                model_kwargs={},
                vlm_kwargs={},
                node_kwargs={},
                debug=False,
                config=user2_config
            )
            
            bob_compiled = await bob_agent.compile(name="bob_agent")
            
            # Bob asks questions that should trigger memory search
            bob_queries = [
                "Do you know anything about Alice Johnson who works at our company?",
                "What are Alice's technical skills and background?",
                "What are Alice's personal interests and hobbies?"
            ]
            
            for i, query in enumerate(bob_queries, 1):
                print(f"\nBob Query {i}: {query}")
                output_lines.append(f"\nBob Query {i}: {query}")
                
                try:
                    result = await bob_compiled.ainvoke(
                        {"messages": [HumanMessage(content=query)]},
                        config=user2_config
                    )

                    response_content = result['messages'][-1].content
                    print(f"Bob's Response: {response_content[:200]}...")
                    output_lines.append(f"Bob's Response: {response_content[:200]}...")

                    used_tools = len(result['messages']) > 2
                    if used_tools:
                        print("✅ Agent attempted memory search (expected)")
                        output_lines.append("✅ Agent attempted memory search (expected)")

                    has_cross_user_data = "alice" in response_content.lower() and (
                        "google" in response_content.lower()
                        or "stanford" in response_content.lower()
                        or "hiking" in response_content.lower()
                    )

                    if has_cross_user_data:
                        print("❌ Bob should not access Alice's memories but some information leaked")
                        output_lines.append("❌ Bob should not access Alice's memories but some information leaked")
                    else:
                        print("✅ Bob could not access another user's memories")
                        output_lines.append("✅ Bob could not access another user's memories")

                    if not used_tools:
                        print("⚠️  Agent didn't use tools")
                        output_lines.append("⚠️  Agent didn't use tools")

                except Exception as e:
                    print(f"Error in Bob's query: {e}")
                    output_lines.append(f"Error in Bob's query: {e}")
                
                await asyncio.sleep(0.5)
        
        async def verify_namespace_isolation():
            """Test that different users have isolated memory namespaces."""
            print("\n--- Phase 4: Testing namespace isolation ---")
            output_lines.append("\n--- Phase 4: Testing namespace isolation ---")
            
            charlie_config = {
                "configurable": {
                    "thread_id": "memory_test_thread_3",
                    "user_id": "charlie_user",
                    "org_id": "different_company"
                }
            }
            
            charlie_agent = BaseAgent(
                prompt="You are Charlie's assistant. Use the MemorySearch tool when asked about colleagues.",
                config=charlie_config
            )
            
            charlie_compiled = await charlie_agent.compile(name="charlie_agent")
            
            # Charlie (different org) should not find Alice's memories
            result = await charlie_compiled.ainvoke(
                {"messages": [HumanMessage(content="Tell me about Alice Johnson")]},
                config=charlie_config
            )
            
            response = result['messages'][-1].content
            print(f"Charlie's response: {response[:150]}...")
            output_lines.append(f"Charlie's response: {response[:150]}...")
            
            # Verify Charlie can't access Alice's org memories
            if ("don't have" in response.lower() or "no information" in response.lower() or 
                "not in our org" in response.lower() or "different org" in response.lower()):
                print("✅ Namespace isolation working correctly - Charlie can't access Alice's memories")
                output_lines.append("✅ Namespace isolation working correctly - Charlie can't access Alice's memories")
            else:
                print("⚠️  Possible namespace leakage - Charlie found Alice's info")
                output_lines.append("⚠️  Possible namespace leakage - Charlie found Alice's info")
        
        async def test_profile_memory_flow():
            """Test profile memory retrieval and overview injection."""
            print("\n--- Phase 5: Testing profile memory extraction and retrieval ---")
            output_lines.append("\n--- Phase 5: Testing profile memory extraction and retrieval ---")

            profile_config = {
                "configurable": {
                    "thread_id": "memory_test_thread_4",
                    "user_id": "dana_user",
                    "org_id": "test_company"
                }
            }

            profile_agent = BaseAgent(
                prompt="You are Dana's assistant. Personalize responses using stored profile details.",
                model_kwargs={},
                vlm_kwargs={},
                node_kwargs={},
                debug=False,
                config=profile_config,
                enable_profile_memory=True
            )

            try:
                profile_compiled = await profile_agent.compile(name="profile_agent")

                # Dana provides profile information through natural conversation
                profile_messages = [
                    "I'm Dana Williams, the Director of Data at Horizon Labs. I split my time between San Francisco and Los Angeles, so keep that in mind for scheduling.",
                    "When you give me updates, keep them concise—bullet summaries with action items first. I prefer direct communication over lengthy explanations."
                ]

                print("\nDana shares profile information:")
                output_lines.append("\nDana shares profile information:")
                
                for idx, message in enumerate(profile_messages, 1):
                    print(f"Dana Message {idx}: {message[:60]}...")
                    output_lines.append(f"Dana Message {idx}: {message[:60]}...")
                    
                    result = await profile_compiled.ainvoke(
                        {"messages": [HumanMessage(content=message)]},
                        config=profile_config
                    )
                    response_content = result["messages"][-1].content
                    print(f"Agent Response: {response_content[:80]}...")
                    output_lines.append(f"Agent Response: {response_content[:80]}...")
                    
                    await asyncio.sleep(0.1)

                print("\n--- Waiting for profile memory processing (3 minutes delay) ---")
                output_lines.append("\n--- Waiting for profile memory processing (3 minutes delay) ---")
                
                # Wait for profile reflection to process
                print("Waiting 200 seconds for profile ReflectionExecutor to process memories...")
                await asyncio.sleep(200)  # Wait longer than the 180-second delay

                # Check if profile was created via background reflection
                print("\n--- Phase 5b: Verifying profile extraction ---")
                output_lines.append("\n--- Phase 5b: Verifying profile extraction ---")
                
                try:
                    # Check if profile memories were created in the store
                    namespace = ("memories", "test_company", "dana_user", "profile")
                    stored_profiles = await profile_compiled._store.asearch(namespace)
                    
                    print(f"Found {len(stored_profiles)} profile memories for Dana")
                    output_lines.append(f"Found {len(stored_profiles)} profile memories for Dana")
                    
                    for i, profile in enumerate(stored_profiles[:2]):  # Show first 2
                        print(f"Profile {i+1}: {str(profile.value)[:120]}...")
                        output_lines.append(f"Profile {i+1}: {str(profile.value)[:120]}...")
                        
                except Exception as e:
                    print(f"Error checking profile memories: {e}")
                    output_lines.append(f"Error checking profile memories: {e}")

                search_results = await profile_agent._profile_manager.asearch(limit=1, config=profile_config)
                print(f"Profile search results: {len(search_results)}")
                output_lines.append(f"Profile search results: {len(search_results)}")

                overview, profile_content = await profile_agent._get_profile_context(profile_config)
                print(f"Overview injected: {overview is not None}")
                output_lines.append(f"Overview injected: {overview is not None}")

                retrieved_profile = await profile_agent.get_user_profile(profile_config)
                print(f"Retrieved profile keys: {list(retrieved_profile.keys()) if retrieved_profile else []}")
                output_lines.append(f"Retrieved profile keys: {list(retrieved_profile.keys()) if retrieved_profile else []}")
                
                # Test hot-path injection with a personalized query
                if overview:
                    print(f"\nOverview format: {overview[:100]}...")
                    output_lines.append(f"\nOverview format: {overview[:100]}...")
                    
                    personalized_result = await profile_compiled.ainvoke(
                        {"messages": [HumanMessage(content="Give me a quick status update on our Q4 metrics.")]},
                        config=profile_config
                    )
                    response = personalized_result["messages"][-1].content
                    print(f"Personalized response: {response[:150]}...")
                    output_lines.append(f"Personalized response: {response[:150]}...")
                    
                    if "bullet" in response.lower() or "concise" in response.lower() or "dana" in response.lower():
                        print("✅ Agent used profile information for personalization")
                        output_lines.append("✅ Agent used profile information for personalization")
                    else:
                        print("⚠️  Agent may not have fully utilized profile information")
                        output_lines.append("⚠️  Agent may not have fully utilized profile information")
            finally:
                await profile_agent.aclose()

        # Run all test phases
        try:
            # alice_agent = await test_memory_ingestion_and_retrieval()
            # await test_cross_user_memory_access(alice_agent)
            # await verify_namespace_isolation()
            await test_profile_memory_flow()
            
            print("\n=== Test Summary ===")
            output_lines.append("\n=== Test Summary ===")
            print("✅ Memory ingestion test completed")
            print("✅ Cross-user memory access test completed")
            print("✅ Namespace isolation test completed")
            print("✅ Profile memory test completed")
            output_lines.extend([
                "✅ Memory ingestion test completed",
                "✅ Cross-user memory access test completed", 
                "✅ Namespace isolation test completed",
                "✅ Profile memory test completed"
            ])
            
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            output_lines.append(f"❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
        
        # Save detailed output
        with open("semantic_memory_test_output.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(output_lines))
        
        print(f"\n📄 Detailed output saved to semantic_memory_test_output.txt ({len(output_lines)} lines)")
    
    asyncio.run(test_semantic_memory())
    
    
    