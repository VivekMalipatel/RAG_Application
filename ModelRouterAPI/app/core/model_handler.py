import logging
from typing import Optional, Dict, List, Union, AsyncGenerator, Any, Type
from pydantic import BaseModel

from openai_client import OpenAIClient
from core.model_provider import Provider
from core.model_type import ModelType
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
        self.logger = logging.getLogger(__name__)
        
        if self.provider == Provider.OPENAI:
            provider_config = kwargs.get('provider_config')
            if not provider_config:
                raise ValueError("provider_config required for OpenAI provider")
            
            self.client = OpenAIClient(
                model_name=model_name,
                api_key=provider_config.get('api_key'),
                base_url=provider_config.get('base_url')
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}.")

    @staticmethod
    async def initialize_from_model_name(
        model_name: str,
        model_type: ModelType,
        **kwargs
    ) -> 'ModelRouter':
        selector = ModelSelector()
        
        try:
            provider, actual_model_name, provider_config = await selector.select_best_model(
                model_type=model_type, 
                model_name=model_name
            )
            
            if provider_config:
                kwargs['provider_config'] = provider_config
            
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

    async def generate_text(self, **kwargs) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        try:
            return await self.client.chat_completions(**kwargs)
        except Exception as e:
            self.logger.error(f"Error generating text with {self.provider.value}: {str(e)}")
            raise

    async def embed(self, **kwargs) -> Dict[str, Any]:
        try:
            return await self.client.embeddings(**kwargs)
        except Exception as e:
            self.logger.error(f"Error generating embeddings with {self.provider.value}: {str(e)}")
            raise

    async def rerank_documents(
        self, 
        query: str, 
        documents: List[str], 
        max_documents: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        raise UnsupportedFeatureError("Document reranking not implemented yet")

    async def generate_structured_output(self, **kwargs) -> Dict[str, Any]:
        try:
            return await self.client.chat_completions(**kwargs)
        except Exception as e:
            self.logger.error(f"Error generating structured output with {self.provider.value}: {str(e)}")
            raise

    def is_model_available(self) -> bool:
        return True
