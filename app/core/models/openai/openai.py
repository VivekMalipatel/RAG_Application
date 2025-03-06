import logging
from typing import Optional, List, Union, AsyncGenerator
from openai import AsyncOpenAI, APIError
from app.config import settings
import asyncio

class OpenAIClient:
    """
    Unified OpenAI client supporting:
    - Text generation (GPT-4, GPT-3.5, DeepSeek)
    - Text embeddings (text-embedding-ada-002)
    - Image generation (DALL·E)
    - Audio transcription (Whisper)
    """

    def __init__(
        self,
        model_name: str = "gpt-4-turbo",
        system_prompt: Optional[str] = "You are a helpful AI assistant.",
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = None,
        stream: bool = False
    ):
        """
        Initializes an OpenAI client with customized model settings.

        Args:
            model_name (str): Model name (e.g., "gpt-4-turbo", "text-embedding-ada-002").
            system_prompt (Optional[str]): Custom system instructions.
            temperature (float): Controls randomness in generation.
            top_p (float): Probability mass for nucleus sampling.
            max_tokens (int): Maximum response length.
            frequency_penalty (float): Penalize repeating words.
            presence_penalty (float): Encourage diversity.
            repetition_penalty (float): Adjust repetition tendency.
            image_quality (str): Image quality setting for DALL·E.
            image_style (str): Image style setting for DALL·E.
        """
        self.client = AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_API_URL
        )
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.stream = stream

        # Ensure the model is available before proceeding
        if not self.is_model_available():
            raise ValueError(f"Model {self.model_name} is not supported by OpenAI.")

    async def generate_text(self, prompt: str, max_tokens: Optional[int] = None, stream: bool = None) -> Union[str, AsyncGenerator[str, None]]:
        """
        Generates a response using OpenAI models (GPT-4, GPT-3.5, DeepSeek).
        """
        stream = stream if stream is not None else self.stream
        
        if stream:
            return self._generate_stream(prompt, max_tokens)
        else:
            return await self._generate_complete(prompt, max_tokens)

    async def _generate_stream(self, prompt: str, max_tokens: Optional[int] = None) -> AsyncGenerator[str, None]:
        """Helper method for streaming text generation."""
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": max_tokens if max_tokens else self.max_tokens,
            "stream": True
        }
        
        try:
            response = await self.client.chat.completions.create(**payload)
            async for chunk in response:
                yield chunk.choices[0].delta.content if chunk.choices[0].delta.content else ""
        except Exception as e:
            logging.error(f"Error: {str(e)}")
            yield f"Error: {str(e)}"

    async def _generate_complete(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """Helper method for non-streaming text generation."""
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": max_tokens if max_tokens else self.max_tokens,
            "stream": False
        }
        
        try:
            response = await self.client.chat.completions.create(**payload)
            return response.choices[0].message.content
        except APIError as e:
            logging.error(f"OpenAI API Error: {str(e)}")
            return f"API Error: {str(e)}"
        except Exception as e:
            logging.error(f"Unexpected Error: {str(e)}")
            return "Error processing request"

    async def get_model_list(self) -> List[str]:
        """
        Retrieves available models from OpenAI.
        """
        try:
            models = await self.client.models.list()
            return [model.id for model in models.data]
        except APIError as e:
            logging.error(f"Model List Error: {str(e)}")
            return []

    def is_model_available(self) -> bool:
        """
        Checks if the requested model is available in OpenAI.
        """
        available_models = asyncio.run(self.get_model_list())
        return self.model_name in available_models

    def set_system_prompt(self, system_prompt: str):
        """Updates the system prompt dynamically."""
        self.system_prompt = system_prompt
        logging.info(f"System prompt updated for OpenAI model {self.model_name}: {system_prompt}")