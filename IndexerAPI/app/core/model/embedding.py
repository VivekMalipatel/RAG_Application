"""
Embedding generation handler for Model Router API.

This module provides utilities for generating text embeddings using
the Model Router API's embedding capabilities.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Union

from app.core.model.api_client import ModelClient

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """Embedding generation handler for creating vector representations of text."""
    
    def __init__(
        self, 
        model_client: Optional[ModelClient] = None,
        model: Optional[str] = None,
        dimensions: Optional[int] = None,
        batch_size: int = 20
    ):
        """
        Initialize the Embedding Generator.
        
        Args:
            model_client: Initialized ModelClient instance
            model: Model identifier to use for embedding generation
            dimensions: Vector dimensions (if None, uses model default)
            batch_size: Number of texts to process in a single batch
        """
        self.client = model_client or ModelClient()
        self.model = model or self.client.config.embedding_model
        self.dimensions = dimensions
        self.batch_size = batch_size
        
        logger.info(f"Initialized Embedding Generator with model: {self.model}")
    
    async def generate_embedding(self, text: str) -> Dict[str, Any]:
        """
        Generate an embedding for a single text.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            Dictionary containing embedding vector and metadata
        """
        try:
            if not text.strip():
                logger.warning("Empty text provided for embedding generation")
                return {
                    "success": False,
                    "error": "Empty text provided",
                    "model": self.model
                }
            
            logger.info(f"Generating embedding for text (length: {len(text)})")
            
            # Create embedding request parameters
            params = {
                "model": self.model,
                "input": text,
            }
            
            if self.dimensions:
                params["dimensions"] = self.dimensions
            
            # Call the Model Router API embedding endpoint
            response = self.client.client.embeddings.create(**params)
            
            # Extract embedding vector and metadata
            embedding = response.data[0].embedding
            
            return {
                "success": True,
                "embedding": embedding,
                "model": self.model,
                "dimensions": len(embedding),
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
                
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "model": self.model
            }
    
    async def generate_embeddings_batch(self, texts: List[str]) -> Dict[str, Any]:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of texts to generate embeddings for
            
        Returns:
            Dictionary containing list of embedding vectors and metadata
        """
        try:
            if not texts:
                logger.warning("Empty text list provided for batch embedding generation")
                return {
                    "success": False,
                    "error": "Empty text list provided",
                    "model": self.model
                }
            
            # Filter out empty texts
            filtered_texts = [text for text in texts if text.strip()]
            
            if not filtered_texts:
                logger.warning("All texts in batch were empty")
                return {
                    "success": False,
                    "error": "All texts in batch were empty",
                    "model": self.model
                }
            
            logger.info(f"Generating embeddings for {len(filtered_texts)} texts")
            
            # Create embedding request parameters
            params = {
                "model": self.model,
                "input": filtered_texts,
            }
            
            if self.dimensions:
                params["dimensions"] = self.dimensions
            
            # Call the Model Router API embedding endpoint
            response = self.client.client.embeddings.create(**params)
            
            # Extract embedding vectors
            embeddings = [data.embedding for data in response.data]
            
            return {
                "success": True,
                "embeddings": embeddings,
                "model": self.model,
                "dimensions": len(embeddings[0]) if embeddings else 0,
                "count": len(embeddings),
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
                
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "model": self.model
            }
    
    async def process_large_text_batch(self, texts: List[str]) -> Dict[str, Any]:
        """
        Process a large batch of texts by splitting into smaller batches.
        
        Args:
            texts: List of texts to generate embeddings for
            
        Returns:
            Dictionary containing list of embedding vectors and metadata
        """
        try:
            if not texts:
                return {
                    "success": False,
                    "error": "Empty text list provided",
                    "model": self.model
                }
            
            # Filter out empty texts
            filtered_texts = [text for text in texts if text.strip()]
            
            total_texts = len(filtered_texts)
            logger.info(f"Processing {total_texts} texts in batches of {self.batch_size}")
            
            all_embeddings = []
            total_tokens = 0
            
            # Process in batches
            for i in range(0, total_texts, self.batch_size):
                batch = filtered_texts[i:i+self.batch_size]
                logger.info(f"Processing batch {i//self.batch_size + 1}/{(total_texts-1)//self.batch_size + 1}")
                
                batch_result = await self.generate_embeddings_batch(batch)
                
                if not batch_result["success"]:
                    logger.error(f"Error processing batch: {batch_result['error']}")
                    continue
                    
                all_embeddings.extend(batch_result["embeddings"])
                total_tokens += batch_result["usage"]["total_tokens"]
            
            return {
                "success": True,
                "embeddings": all_embeddings,
                "model": self.model,
                "dimensions": len(all_embeddings[0]) if all_embeddings else 0,
                "count": len(all_embeddings),
                "usage": {
                    "total_tokens": total_tokens
                }
            }
                
        except Exception as e:
            logger.error(f"Error processing large text batch: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "model": self.model
            }
    
    @staticmethod
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must be of the same length")
            
        # Convert to numpy arrays for efficient computation
        a = np.array(vec1)
        b = np.array(vec2)
        
        # Calculate cosine similarity
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))