import logging
from typing import Tuple

from config import settings
from model_type import ModelType
from model_provider import Provider

logger = logging.getLogger(__name__)

class ModelNotFoundException(Exception):
    pass

class ModelSelectorV2:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        self.model_type_mappings = {
            ModelType.TEXT_GENERATION: settings.TEXT_GENERATION_MODELS,
            ModelType.TEXT_EMBEDDING: settings.TEXT_EMBEDDING_MODELS,
            ModelType.IMAGE_EMBEDDING: settings.IMAGE_EMBEDDING_MODELS,
            ModelType.RERANKER: settings.RERANKER_MODELS
        }

    def _find_provider_for_model(self, model_name: str) -> Provider:
        openai_compatible_config = settings.get_provider_config(model_name)
        if openai_compatible_config:
            return Provider.OPENAI
        
        raise ModelNotFoundException(f"Model '{model_name}' not found in any provider")

    def get_openai_provider_config(self, model_name: str) -> dict:
        return settings.get_provider_config(model_name)

    async def select_best_model(self, 
                              model_type: ModelType, 
                              model_name: str,
                              threshold: float = 0.4) -> Tuple[Provider, str, dict]:
        
        if model_type not in self.model_type_mappings:
            raise ModelNotFoundException(f"Model type '{model_type.value}' is not supported")
        
        available_models = self.model_type_mappings[model_type]
        
        if model_name in available_models:
            provider = self._find_provider_for_model(model_name)
            provider_config = None
            
            if provider == Provider.OPENAI:
                provider_config = self.get_openai_provider_config(model_name)
                if not provider_config:
                    raise ModelNotFoundException(f"No provider configuration found for OpenAI-compatible model '{model_name}'")
            
            return provider, model_name, provider_config
        
        raise ModelNotFoundException(f"Model '{model_name}' not found for type '{model_type.value}'. Available models: {available_models}")
