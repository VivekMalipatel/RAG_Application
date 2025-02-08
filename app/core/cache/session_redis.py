import redis.asyncio as redis
import logging
import asyncio
from app.config import settings

class RedisClient:
    """Manages Redis connection pool and automatic reconnection."""
    
    def __init__(self):
        self.client = None

    async def connect(self):
        """Initialize Redis connection with retries."""
        max_retries = 5
        for attempt in range(max_retries):
            try:
                self.client = await redis.from_url(
                    settings.REDIS_URL, decode_responses=True
                )
                await self.client.ping()
                logging.info("Redis connection established")
                return
            except Exception as e:
                logging.error(f"Redis connection failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise e

    async def close(self):
        """Closes Redis connection pool."""
        if self.client:
            await self.client.close()
            self.client = None
            logging.info("Redis connection closed")

redis_session = RedisClient()

async def get_redis():
    """Yields Redis client for FastAPI dependency injection."""
    await redis_session.connect()
    yield redis_session.client