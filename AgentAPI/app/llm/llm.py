from typing import Any, Sequence, Literal, Union, Optional, Callable, AsyncIterator, List, Dict, cast
import logging

from langchain_core.tools import BaseTool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.callbacks import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
from langchain_core.runnables import Runnable, RunnableConfig, RunnableBinding
from langchain_core.runnables.schema import StreamEvent
from langchain_core.runnables.utils import Output, Input

from config import config as app_config
from llm.utils import VLMProcessor, prepare_input_async, prepare_input_sync
from llm.provider_factory import build_chat_model, normalize_provider


MEDIA_ANNOUNCEMENT = "Analysing Images.....\n\n"

PROVIDER_PARAM_ALLOWLIST = {
    "openai": {"temperature", "top_p", "presence_penalty", "frequency_penalty", "max_tokens"},
    "azure_ai": {"temperature", "top_p", "presence_penalty", "frequency_penalty", "max_tokens", "stop"},
    "bedrock": {"temperature", "max_tokens", "stop_sequences"},
    "google": {"temperature", "top_p", "top_k", "max_tokens"},
}


def _clean_kwargs(values: Dict[str, Any]) -> Dict[str, Any]:
    cleaned: Dict[str, Any] = {}
    for key, value in values.items():
        if value is None:
            continue
        if isinstance(value, str) and value == "":
            continue
        cleaned[key] = value
    return cleaned


def _resolve_provider(override: Optional[str], fallback: Optional[str]) -> str:
    return normalize_provider(override or fallback or "openai")


def _extract_extra(overrides: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = overrides.get(key)
    if isinstance(value, dict):
        return dict(value)
    return {}


def _sampling_defaults(kind: str) -> Dict[str, Any]:
    if kind == "vlm":
        return {
            "temperature": app_config.VLM_LLM_TEMPERATURE,
            "top_p": app_config.VLM_LLM_TOP_P,
            "presence_penalty": app_config.VLM_LLM_PRESENCE_PENALTY,
            "frequency_penalty": app_config.VLM_LLM_REPETITION_PENALTY,
        }
    return {
        "temperature": app_config.REASONING_LLM_TEMPERATURE,
        "top_p": app_config.REASONING_LLM_TOP_P,
        "presence_penalty": app_config.REASONING_LLM_PRESENCE_PENALTY,
        "frequency_penalty": app_config.REASONING_LLM_REPETITION_PENALTY,
    }


def _collect_parameters(kind: str, overrides: Dict[str, Any]) -> Dict[str, Any]:
    defaults = _sampling_defaults(kind)
    params: Dict[str, Any] = {}
    for key, default_value in defaults.items():
        params[key] = overrides.get(key, default_value)
    return params


def _filter_model_parameters(provider: str, params: Dict[str, Any]) -> Dict[str, Any]:
    allowed = PROVIDER_PARAM_ALLOWLIST.get(provider, set())
    filtered: Dict[str, Any] = {}
    for key, value in params.items():
        if key not in allowed:
            continue
        if value is None:
            continue
        filtered[key] = value
    return filtered


def _build_model_kwargs(provider: str, params: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    base = _filter_model_parameters(provider, params)
    extra = _extract_extra(overrides, "model_kwargs")
    merged: Dict[str, Any] = dict(base)
    merged.update(extra)
    return _clean_kwargs(merged)


def _client_kwargs_for_provider(provider: str, overrides: Dict[str, Any]) -> Dict[str, Any]:
    if provider == "openai":
        values = {
            "api_key": overrides.get("api_key", app_config.OPENAI_API_KEY),
            "base_url": overrides.get("base_url", app_config.OPENAI_BASE_URL),
            "timeout": overrides.get("timeout", app_config.LLM_TIMEOUT),
            "max_retries": overrides.get("max_retries", app_config.LLM_MAX_RETRIES),
        }
    elif provider == "azure_ai":
        endpoint = overrides.get("endpoint", overrides.get("azure_endpoint", app_config.AZURE_AI_ENDPOINT))
        credential = overrides.get("credential", overrides.get("azure_credential", app_config.AZURE_AI_CREDENTIAL))
        api_version = overrides.get("api_version", overrides.get("azure_api_version", app_config.AZURE_AI_API_VERSION))
        values = {
            "endpoint": endpoint,
            "credential": credential,
            "api_version": api_version,
        }
    elif provider == "bedrock":
        values = {
            "region_name": overrides.get("region_name", overrides.get("aws_region", app_config.AWS_REGION_NAME)),
            "credentials_profile_name": overrides.get("credentials_profile_name", overrides.get("aws_profile", app_config.AWS_PROFILE)),
            "aws_access_key_id": overrides.get("aws_access_key_id", app_config.AWS_ACCESS_KEY_ID),
            "aws_secret_access_key": overrides.get("aws_secret_access_key", app_config.AWS_SECRET_ACCESS_KEY),
            "aws_session_token": overrides.get("aws_session_token", app_config.AWS_SESSION_TOKEN),
        }
    elif provider == "google":
        values = {
            "google_api_key": overrides.get("google_api_key", overrides.get("api_key", app_config.GOOGLE_API_KEY)),
            "client_options": overrides.get("client_options"),
        }
        if app_config.GOOGLE_API_BASE:
            if not values.get("client_options"):
                values["client_options"] = {}
            if isinstance(values["client_options"], dict):
                values["client_options"].setdefault("api_endpoint", app_config.GOOGLE_API_BASE)
    else:
        values = {}
    values.update(_extract_extra(overrides, "client_kwargs"))
    return _clean_kwargs(values)


class LLM(BaseChatModel):
    reasoning_llm: BaseChatModel
    reasoning_provider: str
    utility_llm: Optional[BaseChatModel]
    utility_provider: str
    reasoning_llm_with_tools: Runnable[Any, Any]
    tools: List[Any]
    _tool_binding_kwargs: Dict[str, Any]
    _bound_binding: Optional[Runnable[Any, Any]]
    vlm_processor: Optional[VLMProcessor]
    vlm_provider: Optional[str]
    logger: logging.Logger

    def __init__(
        self,
        reasoningllm_kwargs: Optional[dict] = None,
        vlm_kwargs: Optional[dict] = None,
        utility_llm_kwargs: Optional[dict] = None,
    ):
        reasoning_overrides = dict(reasoningllm_kwargs or {})
        vlm_overrides = dict(vlm_kwargs or {})
        utility_overrides = dict(utility_llm_kwargs or {})

        reasoning_provider = _resolve_provider(
            reasoning_overrides.get("model_provider"),
            app_config.REASONING_LLM_PROVIDER,
        )
        reasoning_model_name = reasoning_overrides.get("model", app_config.REASONING_LLM_MODEL)
        if not reasoning_model_name:
            raise RuntimeError("Missing REASONING_LLM_MODEL configuration")

        reasoning_params = _collect_parameters("reasoning", reasoning_overrides)
        reasoning_model_kwargs = _build_model_kwargs(reasoning_provider, reasoning_params, reasoning_overrides)
        reasoning_client_kwargs = _client_kwargs_for_provider(reasoning_provider, reasoning_overrides)
        reasoning_llm_instance = build_chat_model(
            reasoning_provider,
            reasoning_model_name,
            client_kwargs=reasoning_client_kwargs,
            model_kwargs=reasoning_model_kwargs,
        )

        utility_provider = _resolve_provider(
            utility_overrides.get("model_provider"),
            app_config.UTIL_LLM_PROVIDER or reasoning_provider,
        )
        utility_model_name = utility_overrides.get(
            "model",
            app_config.UTIL_LLM_MODEL or reasoning_model_name,
        )
        utility_llm_instance: Optional[BaseChatModel]
        if utility_model_name:
            try:
                utility_params = _collect_parameters("utility", utility_overrides)
                utility_model_kwargs = _build_model_kwargs(utility_provider, utility_params, utility_overrides)
                utility_client_kwargs = _client_kwargs_for_provider(utility_provider, utility_overrides)
                utility_llm_instance = build_chat_model(
                    utility_provider,
                    utility_model_name,
                    client_kwargs=utility_client_kwargs,
                    model_kwargs=utility_model_kwargs,
                )
            except Exception as exc:
                logging.getLogger(__name__).warning(
                    "Utility LLM initialization failed; falling back to reasoning model",
                    exc_info=exc,
                )
                utility_llm_instance = None
        else:
            utility_llm_instance = None

        vlm_provider = _resolve_provider(
            vlm_overrides.get("model_provider"),
            app_config.VLM_LLM_PROVIDER,
        )
        vlm_model_name = vlm_overrides.get("model", app_config.VLM_MODEL)
        vlm_llm_instance: Optional[BaseChatModel] = None
        if vlm_model_name:
            try:
                vlm_params = _collect_parameters("vlm", vlm_overrides)
                vlm_model_kwargs = _build_model_kwargs(vlm_provider, vlm_params, vlm_overrides)
                vlm_client_kwargs = _client_kwargs_for_provider(vlm_provider, vlm_overrides)
                vlm_llm_instance = build_chat_model(
                    vlm_provider,
                    vlm_model_name,
                    client_kwargs=vlm_client_kwargs,
                    model_kwargs=vlm_model_kwargs,
                )
            except Exception as exc:
                logging.getLogger(__name__).warning(
                    "VLM initialization failed; media preprocessing disabled",
                    exc_info=exc,
                )
                vlm_llm_instance = None

        vlm_invoke_kwargs = _extract_extra(vlm_overrides, "invoke_kwargs")
        vlm_processor = VLMProcessor(vlm_llm_instance, vlm_invoke_kwargs) if vlm_llm_instance else None

        object.__setattr__(self, "reasoning_llm", reasoning_llm_instance)
        object.__setattr__(self, "reasoning_provider", reasoning_provider)
        object.__setattr__(self, "utility_llm", utility_llm_instance or reasoning_llm_instance)
        object.__setattr__(self, "utility_provider", utility_provider)
        object.__setattr__(self, "vlm_processor", vlm_processor)
        object.__setattr__(self, "vlm_provider", vlm_provider if vlm_llm_instance else None)
        object.__setattr__(self, "tools", [])
        object.__setattr__(self, "_tool_binding_kwargs", {})
        object.__setattr__(self, "_bound_binding", None)
        object.__setattr__(self, "logger", logging.getLogger(__name__))
        object.__setattr__(
            self,
            "reasoning_llm_with_tools",
            cast(Runnable[Any, Any], reasoning_llm_instance),
        )
        object.__setattr__(self, "llm", reasoning_llm_instance)

    def with_structured_output(
        self,
        schema: Union[Dict[str, Any], type],
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> "LLM":
        updated_reasoning_llm = self.reasoning_llm.with_structured_output(
            schema=schema,
            include_raw=include_raw,
            **kwargs,
        )
        object.__setattr__(self, "reasoning_llm", updated_reasoning_llm)
        object.__setattr__(self, "llm", updated_reasoning_llm)
        if self.tools and hasattr(updated_reasoning_llm, "bind_tools"):
            binding_kwargs = dict(self._tool_binding_kwargs)
            tool_choice = binding_kwargs.pop("tool_choice", None)
            if self.reasoning_provider == "bedrock":
                tool_choice = None
            binding = updated_reasoning_llm.bind_tools(
                self.tools,
                tool_choice=tool_choice,
                **binding_kwargs,
            )
            object.__setattr__(self, "reasoning_llm_with_tools", cast(Runnable[Any, Any], binding))
            object.__setattr__(self, "_bound_binding", binding)
        else:
            object.__setattr__(
                self,
                "reasoning_llm_with_tools",
                cast(Runnable[Any, Any], updated_reasoning_llm),
            )
        return self

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], type, Callable, BaseTool]],
        *,
        tool_choice: Optional[Union[str]] = None,
        **kwargs: Any,
    ) -> "LLM":
        object.__setattr__(self, "tools", list(tools))
        if not self.tools or not hasattr(self.reasoning_llm, "bind_tools"):
            object.__setattr__(self, "_tool_binding_kwargs", {})
            object.__setattr__(self, "_bound_binding", None)
            object.__setattr__(
                self,
                "reasoning_llm_with_tools",
                cast(Runnable[Any, Any], self.reasoning_llm),
            )
            return self
        object.__setattr__(self, "_tool_binding_kwargs", {"tool_choice": tool_choice, **kwargs})
        binding_kwargs = dict(self._tool_binding_kwargs)
        tool_choice_value = binding_kwargs.pop("tool_choice", None)
        if self.reasoning_provider == "bedrock":
            tool_choice_value = None
        binding = self.reasoning_llm.bind_tools(
            self.tools,
            tool_choice=tool_choice_value,
            **binding_kwargs,
        )
        object.__setattr__(self, "_bound_binding", binding)
        object.__setattr__(
            self,
            "reasoning_llm_with_tools",
            cast(Runnable[Any, Any], binding),
        )
        return self

    @property
    def bound(self) -> Runnable[Any, Any]:
        binding = getattr(self, "_bound_binding", None)
        target: Runnable[Any, Any]
        if binding is not None:
            target = cast(Runnable[Any, Any], binding)
        else:
            target = self._active_llm()
        if hasattr(target, "bound"):
            return target
        return RunnableBinding(bound=target)

    def _active_llm(self) -> Runnable[Any, Any]:
        return self.reasoning_llm_with_tools

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if app_config.ENABLE_VLM_PREPROCESSING:
            processed_messages = prepare_input_sync(
                messages,
                self.vlm_processor,
                logger=self.logger,
                announcement=MEDIA_ANNOUNCEMENT,
            )
        else:
            processed_messages = messages
        self.logger.info("llm.generate", extra={"message_count": len(processed_messages), "tools_bound": bool(self.tools)})
        if self.tools:
            response = self.reasoning_llm_with_tools.invoke(processed_messages, **kwargs)
            generation = ChatGeneration(message=response)
            return ChatResult(generations=[generation], llm_output=None)
        return self.reasoning_llm._generate(processed_messages, stop=stop, run_manager=run_manager, **kwargs)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if app_config.ENABLE_VLM_PREPROCESSING:
            processed_messages = await prepare_input_async(
                messages,
                self.vlm_processor,
                announcement=MEDIA_ANNOUNCEMENT,
            )
        else:
            processed_messages = messages
        self.logger.info("llm.agenerate", extra={"message_count": len(processed_messages), "tools_bound": bool(self.tools)})
        if self.tools:
            response = await self.reasoning_llm_with_tools.ainvoke(processed_messages, **kwargs)
            generation = ChatGeneration(message=response)
            return ChatResult(generations=[generation], llm_output=None)
        return await self.reasoning_llm._agenerate(processed_messages, stop=stop, run_manager=run_manager, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "multi_modal_ensemble"

    async def ainvoke(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Output:
        if app_config.ENABLE_VLM_PREPROCESSING:
            processed_input = await prepare_input_async(
                input,
                self.vlm_processor,
                announcement=MEDIA_ANNOUNCEMENT,
            )
        else:
            processed_input = input
        target_llm = self._active_llm()
        self.logger.info(
            "llm.ainvoke",
            extra={
                "input_type": type(processed_input).__name__,
                "tools_bound": bool(self.tools),
            },
        )
        return await target_llm.ainvoke(processed_input, config=config, **kwargs)

    async def astream(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> AsyncIterator[Output]:
        if app_config.ENABLE_VLM_PREPROCESSING:
            processed_input = await prepare_input_async(
                input,
                self.vlm_processor,
                announcement=MEDIA_ANNOUNCEMENT,
            )
        else:
            processed_input = input
        target_llm = self._active_llm()
        self.logger.info(
            "llm.astream",
            extra={
                "input_type": type(processed_input).__name__,
                "tools_bound": bool(self.tools),
            },
        )
        async for chunk in target_llm.astream(processed_input, config=config, **kwargs):
            yield chunk

    async def astream_events(
        self,
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
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        if app_config.ENABLE_VLM_PREPROCESSING:
            processed_input = await prepare_input_async(
                input,
                self.vlm_processor,
                announcement=MEDIA_ANNOUNCEMENT,
            )
        else:
            processed_input = input
        target_llm = self._active_llm()
        self.logger.info(
            "llm.astream_events",
            extra={
                "input_type": type(processed_input).__name__,
                "tools_bound": bool(self.tools),
            },
        )
        async for event in target_llm.astream_events(
            processed_input,
            config=config,
            version=version,
            include_names=include_names,
            include_types=include_types,
            include_tags=include_tags,
            exclude_names=exclude_names,
            exclude_types=exclude_types,
            exclude_tags=exclude_tags,
            **kwargs,
        ):
            yield event

    async def abatch(
        self,
        inputs: List[Input],
        config: Optional[Union[RunnableConfig, List[RunnableConfig]]] = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Optional[Any],
    ) -> List[Output]:
        if app_config.ENABLE_VLM_PREPROCESSING:
            processed_inputs: List[Input] = []
            for item in inputs:
                processed_inputs.append(
                    await prepare_input_async(
                        item,
                        self.vlm_processor,
                        announcement=MEDIA_ANNOUNCEMENT,
                    )
                )
        else:
            processed_inputs = inputs
        target_llm = self._active_llm()
        self.logger.info(
            "llm.abatch",
            extra={
                "batch_size": len(processed_inputs),
                "tools_bound": bool(self.tools),
            },
        )
        return await target_llm.abatch(
            processed_inputs,
            config=config,
            return_exceptions=return_exceptions,
            **kwargs,
        )
