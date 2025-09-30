from typing import Any, Sequence, Literal, Union, Optional, Callable, AsyncIterator, List, cast
import typing
from langchain.chat_models import init_chat_model
from langchain_core.tools import BaseTool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.runnables import Runnable, RunnableConfig, RunnableBinding
from langchain_core.runnables.utils import Output, Input
from langchain_core.runnables.schema import StreamEvent
from pydantic import BaseModel
from openai import AsyncOpenAI
from config import config
from typing import (
    Optional
)

from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.callbacks import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun

from llm.utils import (
    prepare_input_async,
    prepare_input_sync,
)
import logging
import asyncio


class LLM(BaseChatModel):
    reasoning_llm: BaseChatModel
    reasoning_llm_with_tools: Runnable[Any, Any]
    vlm_client: AsyncOpenAI
    vlm_model: str
    vlm_model_kwargs: dict[str, Any]
    tools: List[Any]
    _tool_binding_kwargs: dict[str, Any]
    logger: logging.Logger

    def __init__(self, reasoningllm_kwargs: Optional[dict] = None, vlm_kwargs: Optional[dict] = None):
        if reasoningllm_kwargs is None:
            reasoningllm_kwargs = {}
        if vlm_kwargs is None:
            vlm_kwargs = {}
            
        reasoning_llm_filtered_kwargs = {k: v for k, v in reasoningllm_kwargs.items() 
                                         if k not in ["model", "model_provider", "base_url", "api_key"]}
        reasoning_llm_config = {
            "model": reasoningllm_kwargs.get("model", config.REASONING_LLM_MODEL),
            "model_provider": reasoningllm_kwargs.get("model_provider", config.MODEL_PROVIDER),
            "base_url": reasoningllm_kwargs.get("base_url", config.OPENAI_BASE_URL),
            "api_key": reasoningllm_kwargs.get("api_key", config.OPENAI_API_KEY),
            "timeout": reasoningllm_kwargs.get("timeout", config.LLM_TIMEOUT),
            "max_retries": reasoningllm_kwargs.get("max_retries", config.LLM_MAX_RETRIES),
            "temperature": reasoningllm_kwargs.get("temperature", config.REASONING_LLM_TEMPERATURE),
            "top_p": reasoningllm_kwargs.get("top_p", config.REASONING_LLM_TOP_P),
            "presence_penalty": reasoningllm_kwargs.get("presence_penalty", config.REASONING_LLM_PRESENCE_PENALTY),
            **reasoning_llm_filtered_kwargs
        }
        
        vlm_filtered_kwargs = {k: v for k, v in vlm_kwargs.items() 
                              if k not in ["model", "model_provider", "base_url", "api_key", "timeout", "max_retries"]}
        vlm_config = {
            "model": vlm_kwargs.get("model", config.VLM_MODEL),
            "model_provider": vlm_kwargs.get("model_provider", config.MODEL_PROVIDER),
            "base_url": vlm_kwargs.get("base_url", config.OPENAI_BASE_URL),
            "api_key": vlm_kwargs.get("api_key", config.OPENAI_API_KEY),
            "timeout": vlm_kwargs.get("timeout", config.LLM_TIMEOUT),
            "max_retries": vlm_kwargs.get("max_retries", config.LLM_MAX_RETRIES),
            "temperature": vlm_kwargs.get("temperature", config.VLM_LLM_TEMPERATURE),
            "top_p": vlm_kwargs.get("top_p", config.VLM_LLM_TOP_P),
            "top_k": vlm_kwargs.get("top_k", config.VLM_LLM_TOP_K),
            "min_p": vlm_kwargs.get("min_p", config.VLM_LLM_MIN_P),
            "repetition_penalty": vlm_kwargs.get("repetition_penalty", config.VLM_LLM_REPETITION_PENALTY),
            "presence_penalty": vlm_kwargs.get("presence_penalty", config.VLM_LLM_PRESENCE_PENALTY),
            **vlm_filtered_kwargs
        }

        object.__setattr__(self, "reasoning_llm", init_chat_model(**reasoning_llm_config))

        object.__setattr__(
            self,
            "vlm_client",
            AsyncOpenAI(
                api_key=vlm_config["api_key"],
                base_url=vlm_config["base_url"],
                timeout=vlm_config["timeout"],
                max_retries=vlm_config["max_retries"],
            ),
        )
        object.__setattr__(self, "vlm_model", vlm_config["model"])
        object.__setattr__(self, "vlm_model_kwargs", {**vlm_filtered_kwargs})
        object.__setattr__(self, "tools", [])
        object.__setattr__(self, "_tool_binding_kwargs", {})
        object.__setattr__(self, "_bound_binding", None)
        object.__setattr__(self, "logger", logging.getLogger(__name__))
        object.__setattr__(
            self,
            "reasoning_llm_with_tools",
            cast(Runnable[Any, Any], self.reasoning_llm),
        )
    
    def with_structured_output(
        self, schema: Union[dict, type[BaseModel]], **kwargs: Any
    ) -> 'LLM':
        updated_reasoning_llm = self.reasoning_llm.with_structured_output(
            schema=schema,
            **kwargs,
        )
        object.__setattr__(self, "reasoning_llm", updated_reasoning_llm)
        if self.tools:
            binding_kwargs = dict(self._tool_binding_kwargs)
            tool_choice = binding_kwargs.pop("tool_choice", None)
            object.__setattr__(
                self,
                "reasoning_llm_with_tools",
                cast(
                    Runnable[Any, Any],
                    self.reasoning_llm.bind_tools(
                        self.tools,
                        tool_choice=tool_choice,
                        **binding_kwargs,
                    ),
                ),
            )
        else:
            object.__setattr__(
                self,
                "reasoning_llm_with_tools",
                cast(Runnable[Any, Any], self.reasoning_llm),
            )
        return self
    
    def bind_tools(self,
        tools: Sequence[
            Union[typing.Dict[str, Any], type, Callable, BaseTool]
        ],
        *,
        tool_choice: Optional[Union[str]] = None,
        **kwargs: Any,
        ) -> 'LLM' :
        object.__setattr__(self, "tools", list(tools))
        if self.tools:
            object.__setattr__(self, "_tool_binding_kwargs", {"tool_choice": tool_choice, **kwargs})
            binding_kwargs = dict(self._tool_binding_kwargs)
            tool_choice_value = binding_kwargs.pop("tool_choice", None)
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
        else:
            object.__setattr__(self, "_tool_binding_kwargs", {})
            object.__setattr__(self, "_bound_binding", None)
            object.__setattr__(
                self,
                "reasoning_llm_with_tools",
                cast(Runnable[Any, Any], self.reasoning_llm),
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
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        processed_messages = prepare_input_sync(
            messages,
            self.vlm_client,
            self.vlm_model,
            self.vlm_model_kwargs,
            logger=self.logger,
        )
        self.logger.info("llm.generate", extra={"message_count": len(processed_messages), "tools_bound": bool(self.tools)})
        if self.tools:
            response = self.reasoning_llm_with_tools.invoke(processed_messages, **kwargs)
            generation = ChatGeneration(message=response)
            return ChatResult(generations=[generation], llm_output=None)
        return self.reasoning_llm._generate(processed_messages, stop=stop, run_manager=run_manager, **kwargs)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        processed_messages = await prepare_input_async(
            messages,
            self.vlm_client,
            self.vlm_model,
            self.vlm_model_kwargs,
        )
        self.logger.info("llm.agenerate", extra={"message_count": len(processed_messages), "tools_bound": bool(self.tools)})
        if self.tools:
            response = await self.reasoning_llm_with_tools.ainvoke(processed_messages, **kwargs)
            generation = ChatGeneration(message=response)
            return ChatResult(generations=[generation], llm_output=None)
        return await self.reasoning_llm._agenerate(processed_messages, stop=stop, run_manager=run_manager, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "multi_modal_ensemble"

    async def ainvoke(self,
                      input: Input,
                      config: Optional[RunnableConfig] = None,
                      **kwargs: Any
                    )  -> Output :
        processed_input = await prepare_input_async(
            input,
            self.vlm_client,
            self.vlm_model,
            self.vlm_model_kwargs,
        )
        target_llm = self._active_llm()
        self.logger.info(
            "llm.ainvoke",
            extra={
                "input_type": type(processed_input).__name__,
                "tools_bound": bool(self.tools),
            },
        )
        return await target_llm.ainvoke(processed_input, config=config, **kwargs)

    async def astream(self,
                      input: Input,
                      config: Optional[RunnableConfig] = None,
                      **kwargs: Optional[Any]
                    ) -> AsyncIterator[Output]:
        processed_input = await prepare_input_async(
            input,
            self.vlm_client,
            self.vlm_model,
            self.vlm_model_kwargs,
        )
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

    async def astream_events(self,
                             input: Any, 
                             config: Optional[RunnableConfig] = None,
                             *,
                             version: Literal['v1', 'v2'] = "v2", 
                             include_names: Optional[Sequence[str]] = None,
                             include_types: Optional[Sequence[str]] = None,
                             include_tags: Optional[Sequence[str]] = None,
                             exclude_names: Optional[Sequence[str]] = None,
                             exclude_types: Optional[Sequence[str]] = None,
                             exclude_tags: Optional[Sequence[str]] = None,
                             **kwargs: Any) -> AsyncIterator[StreamEvent] :
        processed_input = await prepare_input_async(
            input,
            self.vlm_client,
            self.vlm_model,
            self.vlm_model_kwargs,
        )
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
            **kwargs
        ):
            yield event
    
    async def abatch(self,
                     inputs: list[Input],
                     config: Optional[Union[RunnableConfig, list[RunnableConfig]]] = None,
                     *,
                     return_exceptions: bool = False,
                     **kwargs:  Optional[Any]) -> list[Output] :
        processed_inputs = []
        for item in inputs:
            processed_inputs.append(
                await prepare_input_async(
                    item,
                    self.vlm_client,
                    self.vlm_model,
                    self.vlm_model_kwargs,
                )
            )
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
            **kwargs
        )

llm = LLM()

if __name__ == "__main__":
    async def test_llm():

        test_llm = LLM()

        text_messages = [
            HumanMessage(content="Hello, how are you?")
        ]
        
        media_messages = [
            HumanMessage(content=[
                {"type": "text", "text": "What do you see in this image?"},
                {"type": "image_url", "image_url": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"}}
            ])
        ]
        
        output_lines = []

        thinkingkwargs = {"configurable": {"reasoning_effort": "low"}}

        output_lines.append("=== Testing astream_events with text messages (all parameters) ===")
        try:
            async for event in test_llm.astream_events(
                text_messages,
                config=thinkingkwargs
            ):
                event_str = f"Event: {event}"
                print(event_str)
                output_lines.append(event_str)
        except Exception as e:
            error_str = f"Error with text astream_events: {e}"
            print(error_str)
            output_lines.append(error_str)
        
        output_lines.append("\n=== Testing astream_events with media messages (all parameters) ===")
        try:
            async for event in test_llm.astream_events(
                media_messages
            ):
                event_str = f"Event: {event}"
                print(event_str)
                output_lines.append(event_str)
        except Exception as e:
            error_str = f"Error with media astream_events: {e}"
            print(error_str)
            output_lines.append(error_str)
        
        output_lines.append("\n=== Testing astream_events with all possible output (no filtering) ===")
        try:
            async for event in test_llm.astream_events(
                text_messages,
                config=None,
                version="v2",
                include_names=None,
                include_types=None,
                include_tags=None,
                exclude_names=None,
                exclude_types=None,
                exclude_tags=None
            ):
                event_str = f"All Events: {event}"
                print(event_str)
                output_lines.append(event_str)
        except Exception as e:
            error_str = f"Error with all events astream_events: {e}"
            print(error_str)
            output_lines.append(error_str)
        
        with open("astream_events_output.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(output_lines))
        
        print(f"\nOutput saved to astream_events_output.txt ({len(output_lines)} lines)")
    
    asyncio.run(test_llm())
