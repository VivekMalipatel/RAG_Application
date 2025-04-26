"""
Model Router API handler module.

This module provides an interface to the Model Router API for:
1. Text generation with LLMs
2. Embedding generation
3. Other AI model operations
"""

from app.core.model.api_client import ModelClient
from app.core.model.embedding import EmbeddingGenerator
from app.core.model.text_generation import TextGenerator

__all__ = ["ModelClient", "EmbeddingGenerator", "TextGenerator"]