import logging
import os
import json
from typing import Dict, Any, List, Optional, Union
import numpy as np

from app.core.model.api_client import ModelClient

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    
    def __init__(
        self,
        model: Optional[str] = None,
        api_client: Optional[ModelClient] = None,
        dimensions: Optional[int] = None
    ):
        self.model = model or os.environ.get("DEFAULT_EMBEDDING_MODEL", "text-embedding-ada-002")
        self.dimensions = dimensions
        self.api_client = api_client or ModelClient()
        logger.info(f"Initialized Embedding Generator with model: {self.model}")
    
    async def generate_embedding(
        self,
        text: Union[str, List[str]],
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        if isinstance(text, str):
            input_texts = [text]
            is_single_input = True
        else:
            input_texts = text
            is_single_input = False
        
        logger.info(f"Generating embeddings for {len(input_texts)} text(s) with model: {model or self.model}")
        
        try:
            data = {
                "model": model or self.model,
                "input": input_texts
            }
            
            if self.dimensions:
                data["dimensions"] = self.dimensions
                
            response = await self.api_client._make_request(
                "POST",
                "/embeddings",
                data=data
            )
            
            embeddings_data = response.get("data", [])
            
            if not embeddings_data:
                logger.error("No embeddings in response")
                return {
                    "success": False,
                    "error": "No embeddings returned from API"
                }
            
            embeddings = [item["embedding"] for item in embeddings_data]
            usage = response.get("usage", {})
            
            if is_single_input:
                result = {
                    "success": True,
                    "embedding": embeddings[0],
                    "dimensions": len(embeddings[0]),
                    "usage": usage,
                    "model": model or self.model
                }
            else:
                result = {
                    "success": True,
                    "embeddings": embeddings,
                    "dimensions": len(embeddings[0]) if embeddings else 0,
                    "count": len(embeddings),
                    "usage": usage,
                    "model": model or self.model
                }
                
            return result
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def cosine_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float]
    ) -> float:
        try:
            v1 = np.array(embedding1)
            v2 = np.array(embedding2)
            
            dot_product = np.dot(v1, v2)
            
            mag1 = np.linalg.norm(v1)
            mag2 = np.linalg.norm(v2)
            
            if mag1 > 0 and mag2 > 0:
                return dot_product / (mag1 * mag2)
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {str(e)}")
            return 0.0
    
    async def find_most_similar(
        self,
        query_embedding: List[float],
        embeddings: List[List[float]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        try:
            similarities = []
            for i, emb in enumerate(embeddings):
                similarity = await self.cosine_similarity(query_embedding, emb)
                similarities.append((i, similarity))
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            results = [
                {
                    "index": idx,
                    "similarity": score
                }
                for idx, score in similarities[:top_k]
            ]
            
            return results
            
        except Exception as e:
            logger.error(f"Error finding most similar embeddings: {str(e)}")
            return []