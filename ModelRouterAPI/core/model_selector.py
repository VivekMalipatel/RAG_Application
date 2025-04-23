import logging
from typing import Dict, List, Optional, Tuple, Any

from config import settings
from model_type import ModelType

logger = logging.getLogger(__name__)

class ModelSelector:
    """
    Provides intelligent model routing based on model name patterns.
    Routes to OpenAI for GPT and deepseek models, Ollama for other models,
    and uses Hugging Face for embeddings and rerankers.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def select_provider_for_model(self, model_name: str, model_type: ModelType) -> str:
        """
        Determines the appropriate provider based on the model name and type.
        
        Args:
            model_name: Name of the model
            model_type: Type of model (text generation, embedding, reranker)
            
        Returns:
            str: The provider to use ('openai', 'ollama', or 'huggingface')
        """
        # For embeddings and rerankers, always use Hugging Face
        if model_type == ModelType.TEXT_EMBEDDING or model_type == ModelType.RERANKER:
            return "huggingface"
            
        # For text generation, use name-based routing
        model_lower = model_name.lower()
        
        # Route to OpenAI if model contains 'gpt' or 'deepseek'
        if "gpt" in model_lower or "deepseek" in model_lower:
            return "openai"
            
        # All other models go to Ollama (assuming they're Hugging Face repo names)
        return "ollama"
    
    def select_best_model(
        self, 
        model_type: ModelType,
        preferred_model: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Selects the best model based on model type and preferred model.
        
        Args:
            model_type: The type of model needed (text generation, embedding, reranker)
            preferred_model: Optional preferred specific model
            
        Returns:
            Tuple[str, str]: (selected provider, selected model)
        """
        # If user has specified a preferred model, use that
        if preferred_model:
            provider = self.select_provider_for_model(preferred_model, model_type)
            return provider, preferred_model
        
        # Default models based on type
        if model_type == ModelType.TEXT_EMBEDDING:
            return "huggingface", "sentence-transformers/all-MiniLM-L6-v2"
        elif model_type == ModelType.RERANKER:
            return "huggingface", "BAAI/bge-reranker-base"
        else:  # Text generation
            # Check if OpenAI is available
            if self._is_provider_available("openai"):
                return "openai", "gpt-3.5-turbo"
            # Fallback to Ollama
            elif self._is_provider_available("ollama"):
                return "ollama", "mistral"
            # Last resort
            else:
                return "huggingface", "mistralai/Mistral-7B-Instruct-v0.2"
    
    def _is_provider_available(self, provider: str) -> bool:
        """Check if a provider is configured and available."""
        if provider == "openai":
            return bool(settings.OPENAI_API_KEY)
        elif provider == "huggingface":
            return bool(settings.HUGGINGFACE_API_TOKEN)
        elif provider == "ollama":
            # Ollama is assumed to be available if the URL is set
            return bool(settings.OLLAMA_BASE_URL)
        return False