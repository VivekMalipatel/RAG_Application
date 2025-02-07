import logging
import json
from app.core.cache.session_redis import redis_session

class RedisCache:
    """
    Handles temporary storage of file chunks in Redis 
    using the shared redis session.
    """

    def __init__(self):
        """
        Expects that main.py has already initialized the shared Redis session.
        """
        self.redis = redis_session.client
        if not self.redis:
            logging.error("Redis client is not initialized. Check session_redis connection in main.py.")

    async def set(self, key: str, value: dict, expire: int = 3600):
        """
        Stores a key-value pair in Redis with an expiration time.
        """
        try:
            await self.redis.set(key, json.dumps(value), ex=expire)
            logging.info(f"Stored chunk in Redis: {key}")
        except Exception as e:
            logging.error(f"Error storing chunk in Redis: {e}")

    async def get(self, key: str):
        """
        Retrieves a value from Redis.
        """
        try:
            data = await self.redis.get(key)
            return json.loads(data) if data else None
        except Exception as e:
            logging.error(f"Error retrieving chunk from Redis: {e}")
            return None

    async def delete(self, key: str):
        """
        Deletes a key from Redis.
        """
        try:
            await self.redis.delete(key)
            logging.info(f"Deleted chunk from Redis: {key}")
        except Exception as e:
            logging.error(f"Error deleting chunk from Redis: {e}")