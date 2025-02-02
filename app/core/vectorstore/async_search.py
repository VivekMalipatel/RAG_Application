from app.core.vectorstore.search_engine import SearchEngine
from app.core.models.huggingface.huggingface import HuggingFaceEmbeddingGenerator
from app.core.cache.redis_cache import RedisCache
import asyncio

class AsyncSearch:
    """Performs async search using cached results or Qdrant + BM25."""

    def __init__(self, collection_name="test_collection"):
        self.search_engine = SearchEngine(collection_name)
        self.embedding_generator = HuggingFaceEmbeddingGenerator("BAAI/bge-large-en-v1.5")
        self.cache = RedisCache()

    async def search(self, query: str, top_k=5):
        """Checks cache first, then performs hybrid search if necessary."""
        cached_result = await self.cache.get_cached_result(query)
        if cached_result:
            print("🔹 Returning Cached Result")
            return cached_result

        query_vector = self.embedding_generator.generate_embedding(query)
        results = self.search_engine.search(query, query_vector, top_k)

        await self.cache.cache_result(query, results)
        return results