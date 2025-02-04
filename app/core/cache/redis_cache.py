import aioredis
import logging
import os

class RedisCache:
    """
    Handles temporary storage of file chunks in Redis.
    """

    def __init__(self, redis_url: str):
        """
        Initializes Redis connection.

        Args:
            redis_url (str): Redis connection URL.
        """
        self.redis_url = redis_url
        self.redis = None

    async def connect(self):
        """Establishes a connection to Redis."""
        if not self.redis:
            self.redis = await aioredis.from_url(self.redis_url)
            logging.info("Connected to Redis.")

    async def set(self, key: str, value: bytes, expire: int = 3600):
        """
        Stores a key-value pair in Redis with an expiration time.

        Args:
            key (str): The key to store.
            value (bytes): The file chunk data.
            expire (int, optional): Expiration time in seconds. Defaults to 1 hour.
        """
        try:
            await self.redis.set(key, value, ex=expire)
            logging.info(f"Stored chunk in Redis: {key}")
        except Exception as e:
            logging.error(f"Error storing chunk in Redis: {e}")

    async def get(self, key: str):
        """
        Retrieves a key from Redis.

        Args:
            key (str): The key to retrieve.

        Returns:
            bytes: The file chunk data, or None if not found.
        """
        try:
            data = await self.redis.get(key)
            return data
        except Exception as e:
            logging.error(f"Error retrieving chunk from Redis: {e}")
            return None

    async def delete(self, key: str):
        """
        Deletes a key from Redis.

        Args:
            key (str): The key to delete.
        """
        try:
            await self.redis.delete(key)
            logging.info(f"Deleted chunk from Redis: {key}")
        except Exception as e:
            logging.error(f"Error deleting chunk from Redis: {e}")