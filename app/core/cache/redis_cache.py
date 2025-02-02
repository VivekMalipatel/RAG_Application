import redis
import os
import json
import asyncio

class RedisCache:
    """Handles caching of search results in Redis."""

    def __init__(self):
        self.redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=6379,
            db=0,
            decode_responses=True
        )

    async def cache_result(self, query: str, results):
        """Caches search results in Redis."""
        key = f"search_cache:{query}"
        value = json.dumps(results)
        await asyncio.to_thread(self.redis_client.setex, key, 3600, value)  # Cache for 1 hour

    async def get_cached_result(self, query: str):
        """Retrieves cached results if available."""
        key = f"search_cache:{query}"
        cached_data = await asyncio.to_thread(self.redis_client.get, key)
        return json.loads(cached_data) if cached_data else None