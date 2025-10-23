import os
import typing
import asyncio
import logging
import json
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Sequence, Union, Optional, Callable, AsyncIterator, Literal, Awaitable
from langgraph.config import get_stream_writer
from langgraph.typing import ContextT, InputT, OutputT, StateT
from langgraph.types import (
    All,
    Checkpointer,
    Command,
    StreamMode,
    Durability,
)
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.store.base import BaseStore, IndexConfig
from langgraph.cache.base import BaseCache
from langchain_core.tools import BaseTool
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.schema import StreamEvent
from langchain_core.runnables.utils import Output
from langchain_core.runnables.base import Input
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, RemoveMessage, ToolMessage
from langchain_core.messages.base import messages_to_dict
from langmem import create_manage_memory_tool, create_memory_store_manager
from langmem.reflection import ReflectionExecutor
from langmem.short_term import asummarize_messages
from langmem.short_term.summarization import _preprocess_messages
from openai import BaseModel
from agents.utils import (
    _load_prompt,
    validate_tools,
    build_memory_config,
    coerce_profile_content,
    coerce_procedural_content,
    format_profile_overview,
    format_procedural_overview,
    build_profile_directives,
    build_procedural_directives,
    prepare_profile_precontext_payload,
    wrap_manage_memory_tool_json,
    register_precontext_provider,
    unregister_precontext_provider,
    build_system_precontext,
    make_profile_precontext_provider,
    make_procedural_precontext_provider,
    make_utc_datetime_precontext_provider,
    coerce_running_summary,
    build_message_token_counter,
    count_tokens,
)
from pathlib import Path
from llm.llm import LLM
from llm.utils import prepare_input_async
from embed.embed import JinaEmbeddings
from agents.base_agents.base_state import BaseState
from agents.base_agents.memory.base_checkpointer import BaseMemorySaver
from agents.base_agents.memory.base_store import BaseMemoryStore
from agents.base_agents.memory.base_memorymodels import SemanticMemory, UserProfileMemory, EpisodicMemoryModel, ProceduralMemoryModel
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
        self._episodic_manager = None
        self._episodic_reflection_executor = None
        self._episodic_manage_tool = None
        self._procedural_manager = None
        self._procedural_reflection_executor = None
        self._procedural_manage_tool = None
        self._procedural_memory_key: Optional[str] = None
        self._token_counter: Optional[Callable[[Sequence[Any]], int]] = None
        
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
        self._precontext_providers: dict[str, Callable[[BaseState, RunnableConfig], Awaitable[Any] | Any]] = {}
        register_precontext_provider(self._precontext_providers, "utc_datetime", make_utc_datetime_precontext_provider())

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

    def _register_memory_tool(self, tool: Optional[BaseTool], name: str) -> None:
        if tool is None:
            return
        if hasattr(tool, "name"):
            tool.name = name
        self._memory_tools.append(tool)

    def _get_memory_llm(self):
        util_llm = getattr(self._llm, "utility_llm", None)
        #return util_llm or self._llm.reasoning_llm
        return self._llm.reasoning_llm

    def _setup_semantic_manager(self, store: Optional[BaseStore]):
        self._semantic_manager = None
        self._reflection_executor = None
        self._semantic_manage_tool = None
        if not store:
            return
        namespace = ("memories", "{org_id}", "{user_id}", "semantic")
        semantic_instructions = _load_prompt("semantic_memory_instructions", base_dir=Path(__file__).parent)
        self._semantic_manager = create_memory_store_manager(
            self._get_memory_llm(),
            store=store,
            namespace=namespace,
            schemas=[SemanticMemory],
            instructions=semantic_instructions,
            enable_inserts=True,
            enable_deletes=True,
            query_limit=envconfig.SEMANTIC_MEMORY_QUERY_LIMIT,
        )
        self._reflection_executor = ReflectionExecutor(self._semantic_manager, store=store)
        self._register_memory_tool(self._semantic_manager.search_tool, "search_semantic_memory")
        self._semantic_manage_tool = create_manage_memory_tool(
            namespace=namespace,
            instructions=(
                "Run the semantic memory search tool first to retrieve similar entries. "
                "If the new information supersedes existing memories, delete or update those records instead of creating duplicates. "
                "Only create a new memory when no relevant entry exists, and provide content as JSON matching the semantic memory schema."
            ),
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
        unregister_precontext_provider(self._precontext_providers, "profile_context")
        if not store or not self._config.profile_memory_enabled:
            return
        namespace = ("memories", "{org_id}", "{user_id}", "profile")
        profile_instructions = _load_prompt("profile_memory_instructions", base_dir=Path(__file__).parent)
        self._profile_manager = create_memory_store_manager(
            self._get_memory_llm(),
            store=store,
            namespace=namespace,
            schemas=[UserProfileMemory],
            instructions=profile_instructions,
            enable_inserts=False,
            query_limit=envconfig.PROFILE_MEMORY_QUERY_LIMIT,
        )
        self._profile_reflection_executor = ReflectionExecutor(self._profile_manager, store=store)
        self._register_memory_tool(self._profile_manager.search_tool, "search_profile_memory")
        self._profile_manage_tool = create_manage_memory_tool(
            namespace=namespace,
            instructions=(
                "Fetch the current profile entry (using the provided memory id or the profile search tool) before making changes. "
                "Update that single record in-place, preserving fields the user did not mention. "
                "If the profile must be rewritten, delete the old entry and write one consolidated replacement."
            ),
            schema=UserProfileMemory,
            actions_permitted=("update","delete"),
            store=store,
            name="manage_profile_memory",
        )
        self._profile_manage_tool = wrap_manage_memory_tool_json(
            self._profile_manage_tool,
            tool_name="manage_profile_memory",
        )
        self._memory_tools.append(self._profile_manage_tool)

        def _profile_directives_builder(payload: Any) -> Optional[str]:
            precontext_payload = prepare_profile_precontext_payload(payload)
            if not precontext_payload:
                return None
            return build_profile_directives(precontext_payload)

        profile_provider = make_profile_precontext_provider(
            lambda run_config: self._get_profile_context(run_config),
            _profile_directives_builder,
            lambda: self._profile_memory_key,
        )
        register_precontext_provider(self._precontext_providers, "profile_context", profile_provider)

    def _setup_procedural_manager(self, store: Optional[BaseStore]):
        self._procedural_manager = None
        self._procedural_reflection_executor = None
        self._procedural_manage_tool = None
        self._procedural_memory_key = None
        unregister_precontext_provider(self._precontext_providers, "procedural_context")
        if not store:
            return
        namespace = ("memories", "{org_id}", "{user_id}", "procedural")
        procedural_instructions = _load_prompt("procedural_memory_instructions", base_dir=Path(__file__).parent)
        self._procedural_manager = create_memory_store_manager(
            self._get_memory_llm(),
            store=store,
            namespace=namespace,
            schemas=[ProceduralMemoryModel],
            instructions=procedural_instructions,
            enable_inserts=False,
            query_limit=envconfig.PROCEDURAL_MEMORY_QUERY_LIMIT,
        )
        self._procedural_reflection_executor = ReflectionExecutor(self._procedural_manager, store=store)
        self._register_memory_tool(self._procedural_manager.search_tool, "search_procedural_memory")
        self._procedural_manage_tool = create_manage_memory_tool(
            namespace=namespace,
            instructions=(
                "Retrieve the existing procedural record before applying changes. Update it in place, preserving directives not mentioned. Delete only when replacing it with a consolidated version in the same call."
            ),
            schema=ProceduralMemoryModel,
            actions_permitted=("update","delete"),
            store=store,
            name="manage_procedural_memory",
        )
        self._procedural_manage_tool = wrap_manage_memory_tool_json(
            self._procedural_manage_tool,
            tool_name="manage_procedural_memory",
        )
        self._memory_tools.append(self._procedural_manage_tool)
        procedural_provider = make_procedural_precontext_provider(
            lambda run_config: self._get_procedural_context(run_config),
            build_procedural_directives,
            lambda: self._procedural_memory_key,
        )
        register_precontext_provider(self._precontext_providers, "procedural_context", procedural_provider)

    def _setup_episodic_manager(self, store: Optional[BaseStore]):
        self._episodic_manager = None
        self._episodic_reflection_executor = None
        self._episodic_manage_tool = None
        if not store:
            return
        namespace = ("memories", "{org_id}", "{user_id}", "episodic")
        episodic_instructions = _load_prompt("episodic_memory_instructions", base_dir=Path(__file__).parent)
        self._episodic_manager = create_memory_store_manager(
            self._get_memory_llm(),
            store=store,
            namespace=namespace,
            schemas=[EpisodicMemoryModel],
            instructions=episodic_instructions,
            enable_inserts=True,
            enable_deletes=True,
            query_limit=envconfig.EPISODIC_MEMORY_QUERY_LIMIT,
        )
        self._episodic_reflection_executor = ReflectionExecutor(self._episodic_manager, store=store)
        self._register_memory_tool(self._episodic_manager.search_tool, "search_episodic_memory")
        self._episodic_manage_tool = create_manage_memory_tool(
            namespace=namespace,
            instructions=(
                "Run the episodic memory search tool before modifying entries. Update or delete older episodes when the new experience supersedes them. Provide content as JSON that matches the episodic memory schema."
            ),
            schema=EpisodicMemoryModel,
            store=store,
            name="manage_episodic_memory",
        )
        self._episodic_manage_tool = wrap_manage_memory_tool_json(
            self._episodic_manage_tool,
            tool_name="manage_episodic_memory",
        )
        self._memory_tools.append(self._episodic_manage_tool)

    def _setup_summarizer(self, store: Optional[BaseStore]):
        self._token_counter = None
        model_name = None
        reasoning_llm = getattr(self._llm, "reasoning_llm", None)
        if reasoning_llm is not None:
            model_name = getattr(reasoning_llm, "model_name", None) or getattr(reasoning_llm, "model", None)
        self._token_counter = build_message_token_counter(model_name)

    async def _prepare_messages_for_reflection(self, messages: list[Any]) -> list[Any]:
        vlm_processor = getattr(self._llm, "vlm_processor", None)
        is_configured = False
        if vlm_processor is not None:
            checker = getattr(vlm_processor, "is_configured", None)
            if callable(checker):
                is_configured = checker()
        if not is_configured:
            raise RuntimeError("VLM preprocessing is required for reflection but no VLM configuration is available.")
        
        async def _process_message(message: Any) -> Any:
            if isinstance(message, ToolMessage):
                return await self._process_tool_message_for_reflection(message, vlm_processor)
            return message
        
        processed = await asyncio.gather(*[_process_message(msg) for msg in messages])
        return list(processed)
    
    async def _process_tool_message_for_reflection(self, tool_msg: ToolMessage, vlm_processor: Any) -> ToolMessage:
        content = getattr(tool_msg, "content", None)
        if not isinstance(content, list):
            return tool_msg
        
        try:
            converted_content = await prepare_input_async(
                [HumanMessage(content=content)],
                vlm_processor,
                announcement="",
            )
            if converted_content and len(converted_content) > 0:
                processed_content = getattr(converted_content[0], "content", content)
                return ToolMessage(
                    content=processed_content,
                    tool_call_id=tool_msg.tool_call_id,
                    name=getattr(tool_msg, "name", None),
                    additional_kwargs=getattr(tool_msg, "additional_kwargs", {}),
                )
        except Exception as exc:
            self._logger.error("ToolMessage VLM preprocessing failed", exc_info=exc)
            raise
        
        return tool_msg

    async def summarization_node(self, state: BaseState, config: RunnableConfig) -> dict[str, Any]:
        messages = list(state.get("messages", []))
        run_config = config or self.config
        configurable = run_config.get("configurable", {}) if run_config else {}
        thread_id = configurable.get("thread_id")
        context_payload = dict(state.get("context", {}) or {})
        existing_summary = context_payload.get("running_summary")
        running_summary = coerce_running_summary(existing_summary)
        counter = self._token_counter or build_message_token_counter(None)
        preprocessed = _preprocess_messages(
            messages=messages,
            running_summary=running_summary,
            max_tokens=envconfig.SUMMARIZATION_TARGET_TOKENS,
            max_tokens_before_summary=envconfig.MAX_STATE_TOKENS,
            max_summary_tokens=envconfig.SUMMARIZATION_SUMMARY_TOKENS,
            token_counter=counter,
        )
        should_summarize = bool(preprocessed.messages_to_summarize)
        writer = None
        streaming_enabled = os.getenv("ENABLE_SUMMARIZATION_STREAM_UPDATES", "0") not in {"", "0", "false", "False"}
        if should_summarize and streaming_enabled:
            try:
                writer = get_stream_writer()
                if writer:
                    writer({
                        "type": "summary_status",
                        "stage": "started",
                        "thread_id": thread_id,
                        "messages": len(preprocessed.messages_to_summarize),
                        "tokens": preprocessed.n_tokens_to_summarize,
                    })
            except (RuntimeError, Exception):
                writer = None
        summarization_result = await asummarize_messages(
            messages,
            running_summary=running_summary,
            model=self._llm,
            max_tokens=envconfig.SUMMARIZATION_TARGET_TOKENS,
            max_tokens_before_summary=envconfig.MAX_STATE_TOKENS,
            max_summary_tokens=envconfig.SUMMARIZATION_SUMMARY_TOKENS,
            token_counter=counter,
        )
        updated_summary = summarization_result.running_summary or running_summary
        
        result_messages = list(summarization_result.messages)
        has_non_system_message = any(
            not isinstance(msg, SystemMessage) for msg in result_messages
        )
        
        if should_summarize and not has_non_system_message and messages:
            min_to_retain = envconfig.SUMMARIZATION_MIN_MESSAGES_TO_RETAIN
            recent_messages = messages[-min_to_retain:] if len(messages) > min_to_retain else messages
            non_system_recent = [msg for msg in recent_messages if not isinstance(msg, SystemMessage)]
            
            if non_system_recent:
                result_messages = result_messages + non_system_recent
                self._logger.warning(
                    f"Summarization left only system messages. Retained {len(non_system_recent)} recent non-system messages to maintain conversation flow."
                )
        
        if should_summarize and writer:
            writer({
                "type": "summary_status",
                "stage": "complete",
                "thread_id": thread_id,
                "messages": len(preprocessed.messages_to_summarize),
                "tokens": preprocessed.n_tokens_to_summarize,
                "retained_tokens": count_tokens(result_messages, self._token_counter, logger=self._logger),
            })

        state_update: dict[str, Any] = {}
        if should_summarize:
            state_update["messages"] = [RemoveMessage(id=REMOVE_ALL_MESSAGES)] + result_messages
        
        if updated_summary is not None:
            context_payload["running_summary"] = updated_summary
            state_update["context"] = context_payload
        elif context_payload:
            state_update["context"] = context_payload
            
        return state_update

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
        overview_source = prepare_profile_precontext_payload(content)
        overview_text = format_profile_overview(overview_source)
        return overview_text, content

    async def _get_procedural_context(self, config: Optional[RunnableConfig]) -> tuple[Optional[str], Optional[dict[str, Any]]]:
        self._procedural_memory_key = None
        if not self._procedural_manager or not config:
            return None, None
        try:
            results = await self._procedural_manager.asearch(limit=envconfig.PROCEDURAL_MEMORY_QUERY_LIMIT, config=config)
        except Exception as exc:
            self._logger.error("Procedural snapshot retrieval failed", exc_info=exc)
            return None, None
        if not results:
            return None, None
        item = results[0]
        self._procedural_memory_key = getattr(item, "key", None)
        content = coerce_procedural_content(item.value)
        if not isinstance(content, dict):
            self._procedural_memory_key = None
            return None, None
        overview = format_procedural_overview(content)
        return overview, content

    async def llm_node(self, state: BaseState, config: RunnableConfig):
        context_payload = dict(state.get("context", {}) or {})
        existing_summary = context_payload.get("running_summary")
        running_summary = coerce_running_summary(existing_summary)
        if running_summary is not None:
            context_payload["running_summary"] = running_summary
        else:
            context_payload.pop("running_summary", None)

        base_messages = list(state.get("messages", []))
        token_history = list(state.get("token_usage_history", []) or [])

        state_for_precontext = dict(state)
        state_for_precontext["messages"] = base_messages
        state_for_precontext["context"] = context_payload
        state_for_precontext["token_usage_history"] = token_history

        system_parts = await build_system_precontext(
            self.prompt,
            self._precontext_providers,
            state_for_precontext,
            config,
            fallback_config=self.config,
            logger=self._logger,
        )
        system_messages = [SystemMessage(content="\n\n".join(system_parts))] if system_parts else []
        messages = system_messages + base_messages

        response = await self._llm.ainvoke(messages, config, **self._config.node_kwargs)

        if self._is_structured_output:
            response = AIMessage(content=json.dumps(response))

        messages_updates: list[Any] = [response]

        payload_messages = [*base_messages, response]
        preprocess_for_background = any(
            executor is not None
            for executor in (
                self._reflection_executor,
                self._profile_reflection_executor if self._config.profile_memory_enabled else None,
                self._procedural_reflection_executor,
                self._episodic_reflection_executor,
            )
        )
        reflection_messages = payload_messages
        if preprocess_for_background:
            reflection_messages = await self._prepare_messages_for_reflection(payload_messages)

        usage = getattr(response, "usage_metadata", None)
        counted_tokens = count_tokens(payload_messages, self._token_counter, logger=self._logger)
        token_entry: dict[str, int] = {"approx_tokens": counted_tokens}
        if isinstance(usage, dict):
            for key, value in usage.items():
                if isinstance(value, (int, float)):
                    token_entry[key] = int(value)
        token_history.append(token_entry)
        if len(token_history) > 100:
            token_history = token_history[-100:]

        if self._reflection_executor:
            payload = {
                "messages": reflection_messages,
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
                "messages": reflection_messages,
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
        if self._procedural_reflection_executor:
            procedural_payload = {
                "messages": reflection_messages,
                "max_steps": envconfig.PROCEDURAL_MEMORY_MAX_UPDATES_PER_TURN,
            }
            try:
                self._procedural_reflection_executor.submit(
                    procedural_payload,
                    after_seconds=envconfig.PROCEDURAL_MEMORY_DELAY_SECONDS,
                    config=build_memory_config(config),
                )
            except Exception as exc:
                self._logger.error("Procedural memory submission failed", exc_info=exc)
        if self._episodic_reflection_executor:
            episodic_payload = {
                "messages": reflection_messages,
                "max_steps": envconfig.EPISODIC_MEMORY_MAX_UPDATES_PER_TURN,
            }
            try:
                self._episodic_reflection_executor.submit(
                    episodic_payload,
                    after_seconds=envconfig.EPISODIC_MEMORY_DELAY_SECONDS,
                    config=build_memory_config(config),
                )
            except Exception as exc:
                self._logger.error("Episodic memory submission failed", exc_info=exc)
        result: dict[str, Any] = {
            "messages": messages_updates,
            "token_usage_history": token_history,
        }
        result["context"] = context_payload
        return result

    def _compile_graph(self, has_tools: bool, **compile_kwargs) -> CompiledStateGraph[StateT, ContextT, InputT, OutputT]:
        graph_builder = StateGraph(BaseState)
        llm_node_name = f"llm${self.config.get('configurable').get('user_id')}"
        summarize_node_name = "summarize"
        graph_builder.add_node(summarize_node_name, self.summarization_node)
        graph_builder.add_node(llm_node_name, self.llm_node)
        if has_tools:
            graph_builder.add_node("tools", ToolNode(self._tools))
            graph_builder.add_edge(START, summarize_node_name)
            graph_builder.add_edge(summarize_node_name, llm_node_name)
            graph_builder.add_conditional_edges(
                llm_node_name,
                tools_condition,
                {"tools": "tools", "__end__": END}
            )
            graph_builder.add_edge("tools", summarize_node_name)
        else:
            graph_builder.add_edge(START, summarize_node_name)
            graph_builder.add_edge(summarize_node_name, llm_node_name)
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
        self._setup_procedural_manager(self._store)
        self._setup_episodic_manager(self._store)
        self._setup_summarizer(self._store)
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
        _shutdown_executor("_procedural_reflection_executor")
        _shutdown_executor("_episodic_reflection_executor")

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