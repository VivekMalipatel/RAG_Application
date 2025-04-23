import logging
from typing import Optional, Dict, List, Union, AsyncGenerator

# Import provider-specific clients
from openai import OpenAIClient
from ollama import OllamaClient
from huggingface import HuggingFaceClient

from model_provider import Provider
from model_type import ModelType

class ModelRouter:
    """
    Unified model router to abstract interactions with OpenAI, Ollama, and Hugging Face.
    Supports:
    - OpenAI (GPT-4, GPT-3.5, DeepSeek)
    - Ollama (LLaMA models)
    - Hugging Face (Various transformer models)
    """

    def __init__(
        self,
        provider: Provider,
        model_name: str,
        model_type: ModelType,
        model_quantization: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = 0.7,
        top_p: Optional[float] = 0.9,
        max_tokens: Optional[int] = None,
        stream: Optional[bool] = False,
        **kwargs
    ):
        """
        Initializes the ModelRouter with the appropriate backend.

        Args:
            provider (str): One of ['openai', 'ollama', 'huggingface'].
            model_name (str): The name of the model.
            model_type (str): Task type (text, image, embedding, audio).
            system_prompt (Optional[str]): System prompt for the model.
            temperature (float): Sampling temperature.
            top_p (float): Nucleus sampling threshold.
            max_tokens (int): Maximum response tokens.
            stream (bool): Whether to enable streaming responses.
            frequency_penalty (float): Penalize repeating words.
            presence_penalty (float): Encourage diversity.
            repetition_penalty (float): Adjust repetition tendency.
            top_k (int): Number of top candidates considered.
            num_beams (int): Beam search width.
            kwargs: Additional provider-specific parameters.
        """
        self.provider = provider
        self.model_name = model_name
        self.model_type = model_type
        self.system_prompt = system_prompt
        self.stream = stream
        self.max_tokens = max_tokens

        # Create provider-specific kwargs and initialize the respective client
        if self.provider == Provider.OPENAI:
            openai_kwargs = {
                "model_name": model_name,
                "system_prompt": system_prompt,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "stream": stream,
                **kwargs  # Additional OpenAI-specific params
            }
            self.client = OpenAIClient(**openai_kwargs)

        elif self.provider == Provider.OLLAMA:
            ollama_kwargs = {
                "hf_repo": model_name,
                "system_prompt": system_prompt,
                "quantization": model_quantization,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "stream": stream,
                **kwargs  # Additional Ollama-specific params
            }
            self.client = OllamaClient(**ollama_kwargs)

        elif self.provider == Provider.HUGGINGFACE:
            huggingface_kwargs = {
                "model_name": model_name,
                "model_type": model_type,
                "system_prompt": system_prompt,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "stream": stream,
                **kwargs  # Additional Hugging Face-specific params
            }
            self.client = HuggingFaceClient(**huggingface_kwargs)

        else:
            raise ValueError(f"Unsupported provider: {provider}")

        if not self.is_model_available():
            raise ValueError(f"Model {self.model_name} is not available for provider {self.provider}")
            
    def is_model_available(self) -> bool:
        """
        Checks if the requested model is available in the provider.
        Assumes each model class has an `is_model_available()` method.
        """
        return self.client.is_model_available()

    async def generate_text(self, prompt: str, max_tokens: Optional[int] = None, stream: Optional[bool] = None) -> Union[str, AsyncGenerator[str, None]]:
        """
        Generates a text response.

        Args:
            prompt (str): The user prompt.
            max_tokens (Optional[int]): Maximum number of tokens.
            stream (Optional[bool]): Whether to enable streaming.

        Returns:
            Union[str, AsyncGenerator[str, None]]: The generated text.
        """
        stream = stream if stream is not None else self.stream
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens

        response = await self.client.generate_text(prompt, max_tokens, stream)

        if stream:
            return response
        return response
    
    async def embed_text(self, texts: List[str]) -> List[List[float]]:
        """
        Generates text embeddings.

        Args:
            texts (Union[str, List[str]]): Input text(s).

        Returns:
            List[List[float]]: List of text embeddings.
        """
        return await self.client.embed_text(texts)

    def set_system_prompt(self, system_prompt: str):
        """Updates the system prompt dynamically."""
        self.system_prompt = system_prompt
        self.client.set_system_prompt(system_prompt)
        logging.info(f"System prompt updated for {self.provider}: {system_prompt}")