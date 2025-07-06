from typing import Any, Sequence, Literal, Union, Optional, Callable
import typing
from langchain.chat_models import init_chat_model
from langchain_core.tools import BaseTool
from langchain_core.language_models.chat_models import LanguageModelInput
from config import config
from llm.utils import (
    has_media, process_media_with_vlm
)
import logging
import asyncio
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig


class LLM:
    def __init__(self):
        self.reasoning_llm = init_chat_model(config.REASONING_LLM_MODEL, model_provider=config.MODEL_PROVIDER, base_url=config.OPENAI_BASE_URL, api_key=config.OPENAI_API_KEY)
        self.vlm = init_chat_model(config.VLM_MODEL, model_provider=config.MODEL_PROVIDER, base_url=config.OPENAI_BASE_URL, api_key=config.OPENAI_API_KEY)
        self.tools = []
        self.logger = logging.getLogger(__name__)
        
    def bind_tools(self,
        tools: Sequence[
            Union[typing.Dict[str, Any], type, Callable, BaseTool]
        ],
        *,
        tool_choice: Optional[Union[str]] = None,
        **kwargs: Any,
        ):
        self.tools = tools
        self.reasoning_llm_with_tools = self.reasoning_llm.bind_tools(
            tools, 
            tool_choice=tool_choice, 
            **kwargs
        )
        return self

    async def ainvoke(self,
                      input: LanguageModelInput,
                      config: RunnableConfig | None = None,
                      **kwargs: Any):
        if has_media(input):
            processed_messages = await process_media_with_vlm(self.vlm, input)
        else:
            processed_messages = input

        if self.tools:
            return await self.reasoning_llm_with_tools.ainvoke(processed_messages, config=config, **kwargs)
        else:
            return await self.reasoning_llm.ainvoke(processed_messages, config=config, **kwargs)

    async def astream(self,
                      input: LanguageModelInput,
                      config: RunnableConfig | None = None,
                      **kwargs: Any | None):
        if has_media(input):
            processed_messages = await process_media_with_vlm(self.vlm, input)
        else:
            processed_messages = input
        
        if self.tools:
            async for chunk in self.reasoning_llm_with_tools.astream(processed_messages, config=config, **kwargs):
                yield chunk
        else:
            async for chunk in self.reasoning_llm.astream(processed_messages, config=config, **kwargs):
                yield chunk

    async def astream_events(self,
                             input: Any, 
                             config: RunnableConfig | None = None,
                             *,
                             version: Literal['v1', 'v2'] = "v2", 
                             include_names: Sequence[str] | None = None,
                             include_types: Sequence[str] | None = None,
                             include_tags: Sequence[str] | None = None,
                             exclude_names: Sequence[str] | None = None,
                             exclude_types: Sequence[str] | None = None,
                             exclude_tags: Sequence[str] | None = None,
                             **kwargs: Any):
        if has_media(input):
            processed_messages = await process_media_with_vlm(self.vlm, input)
        else:
            processed_messages = input

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
                     inputs: list[LanguageModelInput],
                     config: RunnableConfig | list[RunnableConfig] | None = None,
                     *,
                     return_exceptions: bool = False,
                     **kwargs: Any | None):
        processed_inputs = []
        for messages in inputs:
            if has_media(messages):
                processed_messages = await process_media_with_vlm(self.vlm, messages)
            else:
                processed_messages = messages
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
        
        print("Testing ainvoke with text-only messages...")
        try:
            response = await test_llm.ainvoke(text_messages)
            print(f"Text response: {response.content}")
        except Exception as e:
            print(f"Error with text messages: {e}")
        
        print("\nTesting ainvoke with media messages...")
        try:
            response = await test_llm.ainvoke(media_messages)
            print(f"Media response: {response.content}")
        except Exception as e:
            print(f"Error with media messages: {e}")
        
        print("\nTesting astream with text messages...")
        try:
            print("Streaming response: ", end="", flush=True)
            async for chunk in test_llm.astream(text_messages):
                if chunk.content:
                    print(chunk.content, end="", flush=True)
            print()
        except Exception as e:
            print(f"Error with streaming: {e}")
        
        print("\nTesting astream with media messages...")
        try:
            print("Streaming media response: ", end="", flush=True)
            async for chunk in test_llm.astream(media_messages):
                if chunk.content:
                    print(chunk.content, end="", flush=True)
            print()
        except Exception as e:
            print(f"Error with media streaming: {e}")
        
        print("\nTesting abatch with multiple messages...")
        try:
            batch_inputs = [text_messages, [HumanMessage(content="What's the weather like?")]]
            responses = await test_llm.abatch(batch_inputs)
            for i, response in enumerate(responses):
                print(f"Batch response {i}: {response.content}")
        except Exception as e:
            print(f"Error with batch: {e}")
    
    asyncio.run(test_llm())

