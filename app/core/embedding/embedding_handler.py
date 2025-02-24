import logging
import asyncio
import torch
from typing import List, Dict, Any, Union
from app.core.models.huggingface.huggingface import HuggingFaceClient
from app.core.models.ollama.ollama import OllamaClient
from app.core.models.openai.openai import OpenAIClient
from app.core.cache.redis_cache import RedisCache
#from app.core.models.imagebind.imagebind_handler import ImageBindClient

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
        model_source: str = "huggingface", 
        model_name: str = None, 
        model_type: str = "text"
    ):
        """
        Initializes the embedding handler with the specified model source.

        Args:
            model_source (str): One of ["huggingface", "ollama", "openai", "imagebind"]
            model_name (str): Model name (e.g., "nomic-ai/nomic-embed-text-v1.5")
            model_type (str): One of ["text", "image", "audio"]
        """
        self.model_source = model_source
        self.model_name = model_name
        self.model_type = model_type
        self.logger = logging.getLogger(__name__)

        # Load the appropriate model
        if model_source == "huggingface":
            self.model = HuggingFaceClient(model_name=model_name, model_type=model_type)
        elif model_source == "ollama":
            self.model = OllamaClient(hf_repo=model_name, embedding=True)
        elif model_source == "openai":
            self.model = OpenAIClient(model_name=model_name, embedding=True)
        #elif model_source == "imagebind":
        #   self.model = ImageBindClient()
        else:
            raise ValueError(f"Unsupported model source: {model_source}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._init_sparse_model()
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
        return f"embedding:{embedding_type}:{self.model_source}:{self.model_name}:{hash_key}"


    def _init_sparse_model(self):
        """Initialize sparse embedding model if enabled."""
        try:
            from rank_bm25 import BM25Okapi
            self.bm25 = BM25Okapi
        except ImportError:
            self.logger.warning("BM25 dependencies not found. Sparse embeddings disabled.")

    async def encode_dense(self, input_data: Union[str, List[str]]) -> List[List[float]]:
        """
        Generate dense embeddings with caching support.
        """
        try:
            # Check cache first
            cache_key = await self._get_cache_key(input_data, "dense")
            cached_result = await self.cache.get(cache_key)
            
            if cached_result:
                self.logger.info("Dense embedding cache hit")
                return cached_result
            
            # If not in cache, compute embeddings
            if isinstance(input_data, str):
                input_data = [input_data]

            result = None
            if self.model_source == "huggingface":
                if self.model_type == "text":
                    result = self.model.encode_text(input_data)
                elif self.model_type == "image":
                    result = self.model.encode_image(input_data)

            elif self.model_source in ["openai", "ollama"]:
                result = await self.model.embed_text(input_data)

            elif self.model_source == "imagebind":
                if self.model_type == "text":
                    result = await asyncio.gather(*[self.model.get_text_embedding(text) for text in input_data])
                elif self.model_type == "image":
                    result = await asyncio.gather(*[self.model.get_image_embedding(image) for image in input_data])
                elif self.model_type == "audio":
                    result = await asyncio.gather(*[self.model.get_audio_embedding(audio) for audio in input_data])

            if result:
                # Convert numpy arrays to lists for JSON serialization
                result = [embedding.tolist() if hasattr(embedding, 'tolist') else embedding for embedding in result]
                # Cache the result
                await self.cache.set(cache_key, result)
                return result
            else:
                raise ValueError("Unsupported model type for embedding.")
        
        except Exception as e:
            self.logger.error(f"Dense embedding failed: {str(e)}")
            return []

    async def encode_sparse(self, text: str) -> Dict[str, float]:
        """
        Generate sparse embeddings using BM25 with caching support.
        """
        try:
            # Check cache first
            cache_key = await self._get_cache_key(text, "sparse")
            cached_result = await self.cache.get(cache_key)
            
            if cached_result:
                self.logger.info("Sparse embedding cache hit")
                return cached_result
            
            # If not in cache, compute embeddings
            tokenized_text = text.split()
            bm25 = self.bm25([tokenized_text])
            scores = bm25.get_scores(tokenized_text)
            result = {token: float(score) for token, score in zip(tokenized_text, scores)}
            
            # Cache the result
            await self.cache.set(cache_key, result)
            return result
            
        except Exception as e:
            self.logger.error(f"Sparse embedding failed: {str(e)}")
            return {}