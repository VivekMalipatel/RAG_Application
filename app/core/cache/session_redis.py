import redis.asyncio as redis
import logging
from app.config import settings

class RedisClient:
    """Manages a Redis connection pool."""
    def __init__(self):
        self.client = None

    async def connect(self):
        """Create a Redis connection pool if not already initialized."""
        if self.client is None:
            self.client = redis.from_url(settings.REDIS_URL, decode_responses=True)
            logging.info("Redis connection pool initialized")

    async def close(self):
        """Closes the Redis connection pool on FastAPI shutdown."""
        if self.client:
            await self.client.close()
            self.client = None
            logging.info("Redis connection closed")

redis_session = RedisClient()

async def get_redis():
    """Yields the Redis client for dependency injection."""
    await redis_session.connect()
    yield redis_session.client