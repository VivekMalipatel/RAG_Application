import logging
from typing import Optional, Dict, List, Union, AsyncGenerator, Any, Type, Tuple
from pydantic import BaseModel

from openai_client import OpenAIClient
from ollama import OllamaClient
from huggingface import HuggingFaceClient

from model_provider import Provider
from model_type import ModelType
from core.model_selector import ModelSelector
from core.model_selector import ModelNotFoundException

class UnsupportedFeatureError(Exception):
    pass

class ModelRouter:
    def __init__(
        self,
        provider: Union[Provider, str],
        model_name: str,
        model_type: ModelType,
        model_quantization: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: Optional[Union[str, List[str]]] = None,
        **kwargs
    ):
        if isinstance(provider, str):
            try:
                self.provider = Provider(provider)
            except ValueError:
                raise ValueError(f"Unsupported provider: {provider}")
        else:
            self.provider = provider
            
        self.model_name = model_name
        self.model_type = model_type
        self.system_prompt = system_prompt
        self.stream = stream
        self.max_tokens = max_tokens
        self.logger = logging.getLogger(__name__)
        
        common_params = {
            "system_prompt": system_prompt,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": stream,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "stop": stop,
            **kwargs
        }
        
        if self.provider == Provider.OPENAI:
            self.client = OpenAIClient(model_name=model_name, **common_params)
            
        elif self.provider == Provider.OLLAMA:
            self.client = OllamaClient(hf_repo=model_name, quantization=model_quantization, **common_params)
            
        elif self.provider == Provider.HUGGINGFACE:
            self.client = HuggingFaceClient(model_name=model_name, model_type=model_type, **common_params)
            
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    @staticmethod
    async def initialize_from_model_name(
        model_name: str,
        model_type: ModelType,
        **kwargs
    ) -> 'ModelRouter':
        selector = ModelSelector()
        
        try:
            provider, actual_model_name = await selector.select_best_model(
                model_type=model_type, 
                model_name=model_name
            )
            
            return ModelRouter(
                provider=provider,
                model_name=actual_model_name,
                model_type=model_type,
                **kwargs
            )
        except ModelNotFoundException as e:
            raise ValueError(f"Model not found: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error initializing model router: {str(e)}")

    async def generate_text(
        self, 
        prompt: Union[str, List], 
        max_tokens: Optional[int] = None, 
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None,
        stream: Optional[bool] = None
    ) -> Union[str, AsyncGenerator[str, None]]:
        try:
            return await self.client.generate_text(
                prompt, 
                max_tokens=max_tokens, 
                temperature=temperature, 
                top_p=top_p,
                stop=stop,
                stream=stream if stream is not None else self.stream
            )
        except Exception as e:
            self.logger.error(f"Error generating text with {self.provider.value}: {str(e)}")
            raise

    async def embed_text(self, texts: List[str]) -> List[List[float]]:
        if not hasattr(self.client, "embed_text"):
            raise UnsupportedFeatureError(f"Provider {self.provider} does not support text embedding")
            
        try:
            return await self.client.embed_text(texts)
        except Exception as e:
            self.logger.error(f"Error generating embeddings with {self.provider.value}: {str(e)}")
            raise
    
    async def embed_image(self, image: List[dict]) -> List[List[float]]:
        if not hasattr(self.client, "embed_image"):
            raise UnsupportedFeatureError(f"Provider {self.provider} does not support image embedding")
            
        try:
            return await self.client.embed_image(image)
        except Exception as e:
            self.logger.error(f"Error generating image embeddings with {self.provider.value}: {str(e)}")
            raise

    async def rerank_documents(
        self, 
        query: str, 
        documents: List[str], 
        max_documents: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        if not hasattr(self.client, "rerank_documents"):
            raise UnsupportedFeatureError(f"Provider {self.provider} does not support document reranking")
            
        try:
            return await self.client.rerank_documents(query, documents, max_documents)
        except Exception as e:
            self.logger.error(f"Error reranking documents with {self.provider.value}: {str(e)}")
            raise

    async def generate_structured_output(
        self,
        prompt: str,
        schema: Union[Dict[str, Any], Type[BaseModel]],
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        if not hasattr(self.client, "generate_structured_output"):
            raise UnsupportedFeatureError(f"Provider {self.provider} does not support structured output generation")
            
        try:
            return await self.client.generate_structured_output(
                prompt=prompt,
                schema=schema,
                max_tokens=max_tokens if max_tokens is not None else self.max_tokens
            )
        except Exception as e:
            self.logger.error(f"Error generating structured output with {self.provider.value}: {str(e)}")
            raise

    async def generate_audio_and_text(
        self, 
        prompt: Union[str, List], 
        max_tokens: Optional[int] = None, 
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None,
        stream: Optional[bool] = None,
        speaker: Optional[str] = "Chelsie",
        use_audio_in_video: bool = True,
        return_audio: bool = True
    ) -> Union[Tuple[str, Any], AsyncGenerator[Tuple[str, Optional[Any]], None]]:
        if not hasattr(self.client, "generate_audio_and_text"):
            raise UnsupportedFeatureError(f"Provider {self.provider} does not support audio generation")
            
        try:
            return await self.client.generate_audio_and_text(
                prompt, 
                max_tokens=max_tokens, 
                temperature=temperature, 
                top_p=top_p,
                stop=stop,
                stream=stream if stream is not None else self.stream,
                speaker=speaker,
                use_audio_in_video=use_audio_in_video,
                return_audio=return_audio
            )
        except Exception as e:
            self.logger.error(f"Error generating audio and text with {self.provider.value}: {str(e)}")
            raise

    def set_system_prompt(self, system_prompt: str) -> None:
        self.system_prompt = system_prompt
        
        if hasattr(self.client, "set_system_prompt"):
            self.client.set_system_prompt(system_prompt)
            
    def is_model_available(self) -> bool:
        if not hasattr(self.client, "is_model_available"):
            return True
            
        try:
            return self.client.is_model_available()
        except Exception as e:
            self.logger.warning(f"Could not check model availability: {str(e)}")
            return False