import redis
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

REDIS_URL = os.getenv("REDIS_URL")

# Initialize Redis client
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)

def cache_set(key: str, value: str, ttl: int = 3600):
    """Store a value in Redis cache with a Time-To-Live (TTL)."""
    redis_client.setex(key, ttl, value)

def cache_get(key: str):
    """Retrieve a cached value from Redis."""
    return redis_client.get(key)

def cache_delete(key: str):
    """Remove a key from Redis."""
    redis_client.delete(key)