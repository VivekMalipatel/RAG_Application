import logging
from typing import Optional, Union, Dict, Any, AsyncGenerator
from openai import AsyncOpenAI
import httpx

class OpenAIClient:
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
        self.base_url=base_url
        self.api_key=api_key
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
            url = f"{self.base_url}/embeddings"
            headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
            }
            kwargs['model'] = self.model_name
            payload = {
                **kwargs
            }
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.post(url, json=payload, headers=headers)
                    response.raise_for_status()
                    return response.json()
            except httpx.HTTPStatusError as e:
                self.logger.error(f"HTTP error in embeddings: {e.response.status_code} - {e.response.text}")
                raise
        except Exception as e:
            self.logger.error(f"Unexpected error in embeddings: {str(e)}")
            raise
