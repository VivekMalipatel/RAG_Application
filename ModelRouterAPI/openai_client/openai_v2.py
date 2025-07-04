import logging
from typing import Optional, Union, Dict, Any, AsyncGenerator
from openai import AsyncOpenAI

class OpenAIClientV2:
    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: str,
        **kwargs
    ):
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)

    async def chat_completions(self, **kwargs) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        try:
            kwargs['model'] = self.model_name
            
            if kwargs.get('stream', False):
                return self._stream_chat_completions(**kwargs)
            else:
                response = await self.client.chat.completions.create(**kwargs)
                return response.model_dump()
        except Exception as e:
            self.logger.error(f"Error in chat completions: {str(e)}")
            raise

    async def _stream_chat_completions(self, **kwargs) -> AsyncGenerator[Dict[str, Any], None]:
        try:
            stream = await self.client.chat.completions.create(**kwargs)
            async for chunk in stream:
                yield chunk.model_dump()
        except Exception as e:
            self.logger.error(f"Error in streaming chat completions: {str(e)}")
            raise

    async def embeddings(self, **kwargs) -> Dict[str, Any]:
        try:
            kwargs['model'] = self.model_name
            response = await self.client.embeddings.create(**kwargs)
            return response.model_dump()
        except Exception as e:
            self.logger.error(f"Error in embeddings: {str(e)}")
            raise
