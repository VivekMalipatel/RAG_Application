import logging
from typing import Optional, Dict, List, Union, AsyncGenerator, Any, Type
from pydantic import BaseModel

# Import provider-specific clients
from openai_client import OpenAIClient
from ollama import OllamaClient
from huggingface import HuggingFaceClient

from model_provider import Provider
from model_type import ModelType
from core.model_selector import ModelSelector

class UnsupportedFeatureError(Exception):
    """Exception raised when a feature is not supported by a provider or model."""
    pass

class ModelRouter:
    """
    Unified model router to abstract interactions with OpenAI, Ollama, and Hugging Face.
    Supports:
    - OpenAI (GPT-4, GPT-3.5, DeepSeek)
    - Ollama (LLaMA and other Hugging Face models)
    - Hugging Face (For embeddings and rerankers)
    """

    def __init__(
        self,
        provider: Union[Provider, str],
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
            provider: Provider enum or string ('openai', 'ollama', 'huggingface').
            model_name: The name of the model.
            model_type: Task type (text generation, embedding, reranker).
            model_quantization: Optional quantization level for Ollama models.
            system_prompt: System prompt for the model.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            max_tokens: Maximum response tokens.
            stream: Whether to enable streaming responses.
            kwargs: Additional provider-specific parameters.
        """
        # Convert string provider to Provider enum if needed
        if isinstance(provider, str):
            try:
                self.provider = Provider(provider)
            except ValueError:
                raise ValueError(f"Unsupported provider string: {provider}")
        else:
            self.provider = provider
            
        self.model_name = model_name
        self.model_type = model_type
        self.system_prompt = system_prompt
        self.stream = stream
        self.max_tokens = max_tokens
        self.logger = logging.getLogger(__name__)

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

        # Verify model availability
        if not self.is_model_available():
            raise ValueError(f"Model {self.model_name} is not available for provider {self.provider}")
            
        # Log successful initialization
        self.logger.info(f"Successfully initialized {self.provider} model: {self.model_name}")
    
    def _check_feature_support(self, feature_name: str) -> bool:
        """
        Check if the current provider/client supports a specific feature.
        
        Args:
            feature_name: Name of the method to check for
            
        Returns:
            bool: True if the feature is supported, False otherwise
        """
        has_feature = hasattr(self.client, feature_name)
        if not has_feature:
            self.logger.warning(f"Feature '{feature_name}' not supported by provider {self.provider}")
        return has_feature
            
    @staticmethod
    def initialize_from_model_name(
        model_name: str,
        model_type: ModelType,
        **kwargs
    ):
        """
        Factory method to initialize a ModelRouter based on model name pattern.
        Uses the ModelSelector to determine the appropriate provider.
        
        Args:
            model_name: Name of the model (e.g., "gpt-4", "mistralai/Mistral-7B")
            model_type: Type of model (text generation, embedding)
            **kwargs: Additional parameters to pass to the ModelRouter constructor
            
        Returns:
            ModelRouter: Configured for the appropriate provider
        """
        selector = ModelSelector()
        provider = selector.select_provider_for_model(model_name, model_type)
        return ModelRouter(
            provider=provider,
            model_name=model_name,
            model_type=model_type,
            **kwargs
        )
            
    def is_model_available(self) -> bool:
        """
        Checks if the requested model is available in the provider.
        Assumes each model class has an `is_model_available()` method.
        """
        if not self._check_feature_support("is_model_available"):
            self.logger.warning(f"Cannot verify if model {self.model_name} is available for {self.provider}")
            return True  # Assume available if we can't check
            
        return self.client.is_model_available()

    async def generate_text(
        self, 
        prompt: str, 
        max_tokens: Optional[int] = None, 
        stream: Optional[bool] = None
    ) -> Union[str, AsyncGenerator[str, None]]:
        """
        Generates a text response.

        Args:
            prompt (str): The user prompt.
            max_tokens (Optional[int]): Maximum number of tokens.
            stream (Optional[bool]): Whether to enable streaming.

        Returns:
            Union[str, AsyncGenerator[str, None]]: The generated text.
            
        Raises:
            UnsupportedFeatureError: If the model doesn't support text generation
        """
        if self.model_type != ModelType.TEXT_GENERATION:
            raise UnsupportedFeatureError(f"Model type {self.model_type} does not support text generation")
            
        if not self._check_feature_support("generate_text"):
            raise UnsupportedFeatureError(f"Provider {self.provider} does not support text generation")

        stream = stream if stream is not None else self.stream
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens

        try:
            response = await self.client.generate_text(prompt, max_tokens, stream)
            return response
        except Exception as e:
            self.logger.error(f"Error generating text with {self.provider}: {str(e)}")
            raise
    
    async def embed_text(self, texts: List[str]) -> List[List[float]]:
        """
        Generates text embeddings.

        Args:
            texts (List[str]): Input texts to embed.

        Returns:
            List[List[float]]: List of text embeddings.
            
        Raises:
            UnsupportedFeatureError: If the model doesn't support embeddings
        """
        if self.model_type != ModelType.EMBEDDING:
            raise UnsupportedFeatureError(f"Model type {self.model_type} does not support embeddings")
            
        if not self._check_feature_support("embed_text"):
            raise UnsupportedFeatureError(f"Provider {self.provider} does not support embeddings")
            
        try:
            return await self.client.embed_text(texts)
        except Exception as e:
            self.logger.error(f"Error generating embeddings with {self.provider}: {str(e)}")
            raise

    async def rerank_documents(
        self, 
        query: str, 
        documents: List[str], 
        max_tokens: int = 512
    ) -> List[int]:
        """
        Reranks documents based on their relevance to a query.
        
        Args:
            query: The search query
            documents: List of document texts to rank
            max_tokens: Maximum tokens to consider for each document
            
        Returns:
            List[int]: Document indices sorted by relevance (highest first)
            
        Raises:
            UnsupportedFeatureError: If the model doesn't support reranking
        """
        if self.model_type != ModelType.RERANKER:
            raise UnsupportedFeatureError(f"Model type {self.model_type} does not support reranking")
            
        if not self._check_feature_support("rerank_documents"):
            raise UnsupportedFeatureError(f"Provider {self.provider} does not support document reranking")
            
        try:    
            return await self.client.rerank_documents(query, documents, max_tokens)
        except Exception as e:
            self.logger.error(f"Error reranking documents with {self.provider}: {str(e)}")
            raise

    def set_system_prompt(self, system_prompt: str):
        """
        Updates the system prompt dynamically.
        
        Raises:
            UnsupportedFeatureError: If the model doesn't support system prompts
        """
        if not self._check_feature_support("set_system_prompt"):
            raise UnsupportedFeatureError(f"Provider {self.provider} does not support setting system prompts")
            
        self.system_prompt = system_prompt
        try:
            self.client.set_system_prompt(system_prompt)
            self.logger.info(f"System prompt updated for {self.provider}: {system_prompt}")
        except Exception as e:
            self.logger.error(f"Failed to update system prompt: {str(e)}")
            raise

    async def generate_structured_output(
        self,
        prompt: str,
        schema: Union[Dict[str, Any], Type[BaseModel]],
        max_tokens: Optional[int] = None,
        stream: Optional[bool] = False
    ) -> Dict[str, Any]:
        """
        Generates a structured JSON response that conforms to the provided schema.

        Args:
            prompt (str): The user prompt.
            schema (Union[Dict[str, Any], Type[BaseModel]]): Either a Pydantic model class or a JSON schema dictionary.
            max_tokens (Optional[int]): Maximum number of tokens.
            stream (Optional[bool]): Whether to enable streaming (always disabled for structured outputs).

        Returns:
            Dict[str, Any]: The generated structured data that conforms to the schema.
            
        Raises:
            UnsupportedFeatureError: If the provider doesn't support structured output
        """
        # Ensure the model type is appropriate for text generation
        if self.model_type != ModelType.TEXT_GENERATION:
            raise UnsupportedFeatureError("Only text generation models support structured output generation")
            
        # Check if the client supports structured output generation
        if not self._check_feature_support("generate_structured_output"):
            raise UnsupportedFeatureError(f"Provider {self.provider} does not support structured output generation")
            
        # Stream is always disabled for structured outputs
        if stream:
            self.logger.warning("Streaming not supported for structured outputs, falling back to non-streaming")
            
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        # Convert Pydantic model to JSON schema if needed
        json_schema = schema
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            json_schema = schema.model_json_schema()
            self.logger.info(f"Converted Pydantic model {schema.__name__} to JSON schema")
        
        try:
            return await self.client.generate_structured_output(
                prompt=prompt,
                schema=json_schema,
                max_tokens=max_tokens,
                stream=False  # Force streaming off
            )
        except Exception as e:
            self.logger.error(f"Error generating structured output with {self.provider}: {str(e)}")
            raise