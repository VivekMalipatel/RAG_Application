from typing import Any, Sequence, Literal, Union, Optional, Callable, AsyncIterator
import typing
from langchain.chat_models import init_chat_model
from langchain.chat_models.base import _ConfigurableModel
from langchain_core.tools import BaseTool
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.chat_models import LanguageModelInput
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.runnables import RunnableConfig, Runnable
from langgraph.config import get_stream_writer
from langchain_core.runnables.utils import Output, Input
from langchain_core.runnables.schema import StreamEvent
from pydantic import BaseModel
from openai import AsyncOpenAI
from config import config
from typing import (
    Optional
)

from llm.utils import (
    has_media, process_media_with_vlm
)
import logging
import asyncio

import os


class LLM:
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

        self.reasoning_llm: BaseChatModel = init_chat_model(**reasoning_llm_config)

        self.vlm_client = AsyncOpenAI(
            api_key=vlm_config["api_key"],
            base_url=vlm_config["base_url"],
            timeout=vlm_config["timeout"],
            max_retries=vlm_config["max_retries"]
        )
        self.vlm_model = vlm_config["model"]
        self.vlm_model_kwargs = {
            # "temperature": vlm_config["temperature"],
            # "top_p": vlm_config["top_p"],
            # "presence_penalty": vlm_config["presence_penalty"],
            # "extra_body":{"top_k": vlm_config["top_k"], "min_p": vlm_config["min_p"]},
            **vlm_filtered_kwargs
        }
        self.tools = []
        self.logger = logging.getLogger(__name__)
    
    def with_structured_output(
        self, schema: Union[dict, type[BaseModel]], **kwargs: Any
    ) -> 'LLM':
        self.reasoning_llm = self.reasoning_llm.with_structured_output(
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
        ) -> 'LLM' :
        self.tools = tools
        self.reasoning_llm_with_tools = self.reasoning_llm.bind_tools(
            tools, 
            tool_choice=tool_choice, 
            **kwargs
        )
        return self

    async def ainvoke(self,
                      input: Input,
                      config: Optional[RunnableConfig] = None,
                      **kwargs: Any
                    )  -> Output :
        if has_media(input):
            writer = get_stream_writer()
            writer(f"Analysing Images.....\n\n")
            processed_messages = await process_media_with_vlm(self.vlm_client, self.vlm_model, input, **self.vlm_model_kwargs)
        else:
            processed_messages = input
        # processed_messages = input
        if self.tools:
            return await self.reasoning_llm_with_tools.ainvoke(processed_messages, config=config, **kwargs)
        else:
            return await self.reasoning_llm.ainvoke(processed_messages, config=config, **kwargs)

    async def astream(self,
                      input: Input,
                      config: Optional[RunnableConfig] = None,
                      **kwargs: Optional[Any]
                    ) -> AsyncIterator[Output]:
        if has_media(input):
            writer = get_stream_writer()
            writer(f"Analysing Images.....\n\n")
            processed_messages = await process_media_with_vlm(self.vlm_client, self.vlm_model, input, **self.vlm_model_kwargs)
        else:
            processed_messages = input
        # processed_messages = input
        if self.tools:
            async for chunk in self.reasoning_llm_with_tools.astream(processed_messages, config=config, **kwargs):
                yield chunk
        else:
            async for chunk in self.reasoning_llm.astream(processed_messages, config=config, **kwargs):
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
        if has_media(input):
            writer = get_stream_writer()
            writer(f"Analysing Images.....\n\n")
            processed_messages = await process_media_with_vlm(self.vlm_client, self.vlm_model, input, **self.vlm_model_kwargs)
        else:
            processed_messages = input
        # processed_messages = input
        if self.tools:
            async for event in self.reasoning_llm_with_tools.astream_events(
                processed_messages, 
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
        else:
            async for event in self.reasoning_llm.astream_events(
                processed_messages, 
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
        for messages in inputs:
            if has_media(messages):
                writer = get_stream_writer()
                writer(f"Analysing Images.....\n\n")                
                processed_messages = await process_media_with_vlm(self.vlm_client, self.vlm_model, messages, **self.vlm_model_kwargs)
            else:
                processed_messages = messages
            # processed_messages = messages
            processed_inputs.append(processed_messages)
        
        if self.tools:
            return await self.reasoning_llm_with_tools.abatch(
                processed_inputs, 
                config=config, 
                return_exceptions=return_exceptions, 
                **kwargs
            )
        else:
            return await self.reasoning_llm.abatch(
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

