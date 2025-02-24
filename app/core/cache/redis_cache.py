import logging
import json
from app.core.cache.session_redis import redis_session
import asyncio
import hashlib

class RedisCache:
    """
    Handles temporary storage of file chunks in Redis
    using the shared redis session.
    """

    def __init__(self):
        """Expects that main.py has already initialized the shared Redis session."""
        self.redis = redis_session.client
        if not self.redis:
            logging.error("Redis client is not initialized. Check session_redis connection.")

    async def set(self, key: str, value: dict, expire: int = 3600):
        """
        Stores a key-value pair in Redis with an expiration time.
        Retries if Redis is unavailable.
        """
        retries = 3
        for attempt in range(retries):
            try:
                if not self.redis:
                    await redis_session.connect()
                    self.redis = redis_session.client
                await self.redis.set(key, json.dumps(value), ex=expire)
                logging.info(f"Stored chunk in Redis: {key}")
                return
            except Exception as e:
                logging.error(f"Redis SET failed (Attempt {attempt+1}): {e}")
                await asyncio.sleep(2 ** attempt)
        raise Exception("Redis SET failed after retries.")

    async def get(self, key: str):
        """Retrieves a value from Redis with auto-reconnect."""
        try:
            if not self.redis:
                await redis_session.connect()
                self.redis = redis_session.client
            data = await self.redis.get(key)
            return json.loads(data) if data else None
        except Exception as e:
            logging.error(f"Redis GET failed: {e}")
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
    
    async def get_hash(self, text):
        """Returns a unique hash for a given document text."""
        return hashlib.sha256(text.encode()).hexdigest()
    
    async def purge_cache(self):
        """
        Purges all keys from the Redis cache.
        Returns the number of keys that were removed.
        """
        try:
            if not self.redis:
                await redis_session.connect()
                self.redis = redis_session.client
            
            # Get count of keys before deletion
            keys_count = await self.redis.dbsize()
            
            # Delete all keys in the current database
            await self.redis.flushdb()
            
            logging.info(f"Successfully purged {keys_count} keys from Redis cache")
            return keys_count
            
        except Exception as e:
            logging.error(f"Failed to purge Redis cache: {e}")
            raise Exception(f"Cache purge failed: {e}")