import logging
import asyncio
import time
from typing import Dict, List, Optional, Tuple, Any
from difflib import SequenceMatcher

from config import settings
from model_type import ModelType
from model_provider import Provider
from openai_client import OpenAIClient
from ollama.model_loader import OllamaModelLoader

logger = logging.getLogger(__name__)

class ModelNotFoundException(Exception):
    pass

class ModelSelector:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._openai_models = []
        self._ollama_models = []
        self._last_refresh = 0
        self._cache_ttl = 300

    async def refresh_models(self) -> Dict[str, List]:
        current_time = time.time()
        
        if current_time - self._last_refresh < self._cache_ttl:
            return {
                "openai": self._openai_models,
                "ollama": self._ollama_models,
                "huggingface": []
            }
        
        tasks = [
            self._fetch_openai_models(),
            self._fetch_ollama_models()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        if not isinstance(results[0], Exception):
            self._openai_models = results[0]
        if not isinstance(results[1], Exception):
            self._ollama_models = results[1]
        
        self._last_refresh = current_time
        
        return {
            "openai": self._openai_models,
            "ollama": self._ollama_models,
            "huggingface": []
        }

    async def _fetch_openai_models(self) -> List[str]:
        if not settings.OPENAI_API_KEY:
            return []
            
        try:
            client = OpenAIClient()
            if hasattr(client, "get_model_list"):
                return await client.get_model_list()
        except Exception as e:
            self.logger.error(f"Error fetching OpenAI models: {str(e)}")
            return []

    async def _fetch_ollama_models(self) -> List[str]:
        if not settings.OLLAMA_BASE_URL:
            return []
            
        try:
            loader = OllamaModelLoader()
            models = await loader.get_available_models()
            return [model.get("name", "").split(":")[0] for model in models]
        except Exception as e:
            self.logger.error(f"Error fetching Ollama models: {str(e)}")
            return []

    def _similarity_score(self, str1: str, str2: str) -> float:
        s1 = str1.lower().replace("-", "").replace("_", "").replace(" ", "")
        s2 = str2.lower().replace("-", "").replace("_", "").replace(" ", "")
        
        return SequenceMatcher(None, s1, s2).ratio()

    def _get_best_matching_model(self, model_name: str, available_models: List[str], 
                               threshold: float = 0.6) -> Tuple[str, float]:
        if not available_models:
            return "", 0.0
            
        for model in available_models:
            if model.lower() == model_name.lower():
                return model, 1.0
                
        scores = [(model, self._similarity_score(model_name, model)) for model in available_models]
        
        scores.sort(key=lambda x: x[1], reverse=True)
        
        if scores and scores[0][1] >= threshold:
            return scores[0]
        
        return "", 0.0

    async def select_best_model(self, 
                              model_type: ModelType, 
                              model_name: str,
                              threshold: float = 0.4) -> Tuple[Provider, str]:
        if model_type == ModelType.TEXT_EMBEDDING:
            return Provider.HUGGINGFACE, "nomic-ai/colnomic-embed-multimodal-3b"
        
        if model_type == ModelType.IMAGE_EMBEDDING:
            return Provider.HUGGINGFACE, "nomic-ai/colnomic-embed-multimodal-3b"
        
        if model_type == ModelType.RERANKER:
            return Provider.HUGGINGFACE, "jinaai/jina-colbert-v2"
        
        await self.refresh_models()
        
        openai_match, openai_score = self._get_best_matching_model(
            model_name, self._openai_models, threshold
        )
        
        ollama_match, ollama_score = self._get_best_matching_model(
            model_name, self._ollama_models, threshold
        )
        
        if openai_score >= threshold and openai_score >= ollama_score:
            if len(openai_match)<len(model_name):
                openai_match = model_name
            return Provider.OPENAI, openai_match
        
        if ollama_score >= threshold:
            if len(ollama_match)<len(model_name):
                ollama_match = model_name
            return Provider.OLLAMA, ollama_match
            
        raise ModelNotFoundException(f"Could not find a matching model for '{model_name}' of type {model_type.value}")
        
    def select_provider_for_model(self, model_name: str, model_type: ModelType) -> Provider:
        model_name_lower = model_name.lower()
        
        if model_type == ModelType.TEXT_EMBEDDING:
            return Provider.HUGGINGFACE
        
        if model_type == ModelType.IMAGE_EMBEDDING:
            return Provider.HUGGINGFACE
                    
        if model_type == ModelType.RERANKER:
            return Provider.HUGGINGFACE
        
        openai_patterns = ["gpt-", "text-davinci", "text-embedding", "dall-e", "claude", "command-r"]
        if any(pattern in model_name_lower for pattern in openai_patterns):
            return Provider.OPENAI
            
        ollama_patterns = ["llama", "mistral", "mixtral", "phi", "falcon", "gemma"]
        if any(pattern in model_name_lower for pattern in ollama_patterns):
            return Provider.OLLAMA
            
        if "/" in model_name:
            return Provider.HUGGINGFACE
            
        return Provider.OPENAI