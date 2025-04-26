"""
Base client for interacting with the Model Router API.

This module provides a client for communicating with the Model Router API,
which is compatible with the OpenAI API structure.
"""

import logging
import os
from typing import Dict, Any, Optional, Union, List
import openai
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class ModelConfig(BaseModel):
    """Configuration for Model Router API connection."""
    api_key: str = "test-key"  # Many clients require an API key, even if not used
    base_url: str = "http://localhost:8000/v1"
    timeout: int = 120  # seconds
    default_model: str = "gpt-3.5-turbo"  # Default model identifier
    embedding_model: str = "text-embedding-ada-002"  # Default embedding model

class ModelClient:
    """Client for interacting with the Model Router API."""
    
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        base_url: Optional[str] = None,
        timeout: Optional[int] = None,
        default_model: Optional[str] = None,
        embedding_model: Optional[str] = None
    ):
        """
        Initialize the Model Client.
        
        Args:
            api_key: API key for authentication (optional for local Model Router)
            base_url: Base URL of the Model Router API
            timeout: Request timeout in seconds
            default_model: Default model to use for text generation
            embedding_model: Default model to use for embeddings
        """
        # Load configuration from environment variables or use defaults
        self.config = ModelConfig(
            api_key=api_key or os.getenv("MODEL_ROUTER_API_KEY", "test-key"),
            base_url=base_url or os.getenv("MODEL_ROUTER_BASE_URL", "http://localhost:8000/v1"),
            timeout=timeout or int(os.getenv("MODEL_ROUTER_TIMEOUT", "120")),
            default_model=default_model or os.getenv("MODEL_ROUTER_DEFAULT_MODEL", "gpt-3.5-turbo"),
            embedding_model=embedding_model or os.getenv("MODEL_ROUTER_EMBEDDING_MODEL", "text-embedding-ada-002")
        )
        
        # Configure the OpenAI client with our Model Router API settings
        self.client = openai.OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout
        )
        
        logger.info(f"Initialized Model Client with base URL: {self.config.base_url}")
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get list of available models from the Model Router API.
        
        Returns:
            List of model information dictionaries
        """
        try:
            response = self.client.models.list()
            logger.info(f"Retrieved {len(response.data)} models from Model Router API")
            return [model.model_dump() for model in response.data]
        except Exception as e:
            logger.error(f"Error retrieving models: {str(e)}")
            return []
    
    def health_check(self) -> bool:
        """
        Check if the Model Router API is available.
        
        Returns:
            True if the API is available, False otherwise
        """
        try:
            # Most OpenAI-compatible APIs have a /models endpoint that can be used as a health check
            _ = self.client.models.list()
            return True
        except Exception as e:
            logger.error(f"Model Router API health check failed: {str(e)}")
            return False