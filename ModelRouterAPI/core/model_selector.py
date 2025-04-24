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
    """Exception raised when a model cannot be found by the selector"""
    pass

class ModelSelector:
    """Provides intelligent model routing based on available models"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._openai_models = []
        self._ollama_models = []
        self._last_refresh = 0
        self._cache_ttl = 300  # Cache for 5 minutes

    async def refresh_models(self) -> Dict[str, List]:
        """Refresh available models from all providers"""
        current_time = time.time()
        
        # Skip refresh if cache is still fresh
        if current_time - self._last_refresh < self._cache_ttl:
            return {
                "openai": self._openai_models,
                "ollama": self._ollama_models,
                "huggingface": []  # HF models aren't fetched dynamically
            }
        
        # Fetch models in parallel
        tasks = [
            self._fetch_openai_models(),
            self._fetch_ollama_models()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
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
        """Fetch available models from OpenAI"""
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
        """Fetch available models from Ollama"""
        if not settings.OLLAMA_BASE_URL:
            return []
            
        try:
            loader = OllamaModelLoader()
            models = await loader.get_available_models()
            return [model.get("name", "").split(":")[0] for model in models]  # Remove tags
        except Exception as e:
            self.logger.error(f"Error fetching Ollama models: {str(e)}")
            return []

    def _similarity_score(self, str1: str, str2: str) -> float:
        """Calculate similarity between two strings using SequenceMatcher"""
        # Normalize strings for better matching
        s1 = str1.lower().replace("-", "").replace("_", "").replace(" ", "")
        s2 = str2.lower().replace("-", "").replace("_", "").replace(" ", "")
        
        # Use sequence matcher for string similarity
        return SequenceMatcher(None, s1, s2).ratio()

    def _get_best_matching_model(self, model_name: str, available_models: List[str], 
                               threshold: float = 0.6) -> Tuple[str, float]:
        """Find the best matching model from available models"""
        if not available_models:
            return "", 0.0
            
        # Check for exact match first
        for model in available_models:
            if model.lower() == model_name.lower():
                return model, 1.0
                
        # Calculate similarity scores for each model
        scores = [(model, self._similarity_score(model_name, model)) for model in available_models]
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return best match if it exceeds threshold
        if scores and scores[0][1] >= threshold:
            return scores[0]
        
        return "", 0.0

    async def select_best_model(self, 
                              model_type: ModelType, 
                              model_name: str,
                              threshold: float = 0.4) -> Tuple[Provider, str]:
        """
        Select the best model from available providers based on type and name.
        Returns the provider and the actual model name to use.
        
        Args:
            model_type: Type of model needed
            model_name: Requested model name (can be partial)
            threshold: Minimum similarity threshold for a match
            
        Returns:
            Tuple of (Provider, complete_model_name)
            
        Raises:
            ModelNotFoundException: If no suitable model is found
        """
        # For embedding and reranker types, we use hardcoded Hugging Face models
        if model_type == ModelType.TEXT_EMBEDDING:
            return Provider.HUGGINGFACE, "nomic-ai/colnomic-embed-multimodal-7b"
        
        if model_type == ModelType.RERANKER:
            return Provider.HUGGINGFACE, "jinaai/jina-colbert-v2"
        
        # For text generation models, try to find a match from available models
        # Refresh available models from providers
        await self.refresh_models()
        
        # Try to find a match in OpenAI models first
        openai_match, openai_score = self._get_best_matching_model(
            model_name, self._openai_models, threshold
        )
        
        # Try to find a match in Ollama models
        ollama_match, ollama_score = self._get_best_matching_model(
            model_name, self._ollama_models, threshold
        )
        
        # Determine the best match between providers
        if openai_score >= threshold and openai_score >= ollama_score:
            return Provider.OPENAI, openai_match
        
        if ollama_score >= threshold:
            return Provider.OLLAMA, ollama_match
            
        raise ModelNotFoundException(f"Could not find a matching model for '{model_name}' of type {model_type.value}")
        
    # Keep the old method for backward compatibility
    def select_provider_for_model(self, model_name: str, model_type: ModelType) -> Provider:
        """
        Determine the appropriate provider based on model name pattern
        
        Args:
            model_name: Name of the model
            model_type: Type of model (text generation, embedding, etc.)
            
        Returns:
            Provider enum representing the selected provider
        """
        model_name_lower = model_name.lower()
        
        # Known embedding and reranker models
        if model_type == ModelType.TEXT_EMBEDDING:
            return Provider.HUGGINGFACE
                    
        if model_type == ModelType.RERANKER:
            return Provider.HUGGINGFACE
        
        # Check for OpenAI model patterns
        openai_patterns = ["gpt-", "text-davinci", "text-embedding", "dall-e", "claude", "command-r"]
        if any(pattern in model_name_lower for pattern in openai_patterns):
            return Provider.OPENAI
            
        # Check for Ollama model patterns
        ollama_patterns = ["llama", "mistral", "mixtral", "phi", "falcon", "gemma"]
        if any(pattern in model_name_lower for pattern in ollama_patterns):
            return Provider.OLLAMA
            
        # Check for Hugging Face model patterns (with organization/model format)
        if "/" in model_name:
            return Provider.HUGGINGFACE
            
        # Default to OpenAI if we can't determine
        return Provider.OPENAI