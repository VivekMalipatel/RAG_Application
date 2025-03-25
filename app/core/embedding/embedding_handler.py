import logging
import torch
from typing import List, Dict, Any, Union
from fastembed import SparseTextEmbedding
from qdrant_client.http.models import SparseVector
from app.core.models.model_handler import ModelRouter
from app.core.models.model_provider import Provider
from app.core.models.model_type import ModelType
from app.core.cache.redis_cache import RedisCache
import numpy as np
import json

class EmbeddingHandler:
    """
    Unified embedding handler for multimodal embeddings.
    Supports:
    - Hugging Face (text, image)
    - OpenAI (text)
    - Ollama (text)
    - ImageBind (text, image, audio)
    - Sparse embeddings (BM25)
    """

    def __init__(
        self, 
        provider: Provider = Provider.HUGGINGFACE,
        model_name: str = None, 
        model_type: ModelType = ModelType.TEXT_EMBEDDING
    ):
        """
        Initializes the embedding handler with the specified model source.

        Args:
            provider (str): One of ["huggingface", "ollama", "openai"]
            model_name (str): Model name (e.g., "nomic-ai/nomic-embed-text-v1.5")
            model_type (str): One of ["text_embedding", "image_embedding"]
        """
        self.provider = provider
        self.model_name = model_name
        self.model_type = model_type
        self.sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
        self.logger = logging.getLogger(__name__)

        # Initialize ModelRouter dynamically
        self.model = ModelRouter(
            provider=provider,
            model_name=model_name,
            model_type=model_type
        )
        self.cache = RedisCache()
    
    async def _get_cache_key(self, input_data: Union[str, List[str]], embedding_type: str) -> str:
        """
        Generate a unique cache key for embeddings.
        
        Args:
            input_data: Input text or list of texts
            embedding_type: Either 'dense' or 'sparse'
            
        Returns:
            str: Cache key
        """
        if isinstance(input_data, list):
            input_str = "_".join(input_data)
        else:
            input_str = input_data
            
        hash_key = await self.cache.get_hash(input_str)
        return f"embedding:{embedding_type}:{self.provider}:{self.model_name}:{hash_key}"

    async def encode_dense(self, input_data: Union[str, List[str]]) -> List :
        """
        Generates dense embeddings and caches them.
        """
        try:
            cache_key = await self._get_cache_key(input_data, "dense")
            cached_result = await self.cache.get(cache_key)
            if cached_result:
                self.logger.info("Dense embedding cache hit")
                return json.loads(cached_result)

            # Compute new embeddings
            if isinstance(input_data, str):
                input_data = [input_data]

            result = await self.model.embed_text(input_data)

            if not result:
                raise ValueError("Embedding model returned empty result.")
            
            # Cache the result
            await self.cache.set(cache_key, json.dumps(result))

            return result

        except Exception as e:
            self.logger.error(f"Dense embedding failed: {str(e)}")
            return []

    #TODO : Add Support for Batches (same as dense)
    async def encode_sparse(self, text: str) -> Dict[str, Any]:
        """
        Generates a sparse vector using BM25.
        - Checks cache first and returns if available.
        - If not cached, generates a new sparse vector.
        - Stores the generated vector in cache (converted to JSON).
        - Returns the original format for immediate use.
        """
        try:
            cache_key = await self._get_cache_key(text, "sparse")
            cached_result = await self.cache.get(cache_key)

            if cached_result:
                self.logger.info("Sparse embedding cache hit")

                # Convert JSON back to NumPy array format
                cached_data = json.loads(cached_result)
                embeddings = SparseVector(**cached_data)
                
                return embeddings

            # Compute new sparse embedding
            embeddings = list(self.sparse_model.embed([text]))[0]

            # Extract indices and values
            nonzero_indices = embeddings.indices.tolist()
            nonzero_values = embeddings.values.tolist()

            # Convert to JSON-storable format
            sparse_vector = {
                "indices": nonzero_indices,
                "values": nonzero_values,
            }

            # Store in cache
            await self.cache.set(cache_key, json.dumps(sparse_vector))

            return SparseVector(**sparse_vector)

        except Exception as e:
            self.logger.error(f"Sparse embedding failed: {str(e)}")
            return {"indices": [], "values": []}