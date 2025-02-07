from fastapi import FastAPI
from app.api.v1.endpoints import agent, documents, user
from app.core.cache.redis_cache import RedisCache
from app.core.storage_bin.minio import MinIOHandler
from app.config import settings
from app.api.db.base import get_db
import redis.asyncio as aioredis  # new import

async def lifespan(app: FastAPI):
    MINIO_CONFIG = {
        "endpoint": settings.MINIO_ENDPOINT,
        "access_key": settings.MINIO_ACCESS_KEY,
        "secret_key": settings.MINIO_SECRET_KEY,
    }
    app.state.minio = MinIOHandler(**MINIO_CONFIG)
    # Create and store the redis connection in app.state, then inject it into RedisCache.
    redis_conn = await aioredis.from_url(settings.REDIS_URL)
    app.state.redis = RedisCache(redis_conn)
    await get_db()
    yield
    # Shutdown: add shutdown logic if needed
    if hasattr(app.state.minio, "close"):
        await app.state.minio.close()
    if hasattr(app.state.redis, "close"):
        await app.state.redis.close()

# Create FastAPI instance with lifespan events
app = FastAPI(
    title="OmniRAG API",
    description="API for OmniRAG's document processing and retrieval",
    version="1.0.0",
    lifespan=lifespan
)

# Include API Routers
app.include_router(user.router, prefix="/users", tags=["Users"])
app.include_router(agent.router, prefix="/agents", tags=["Agents"])
app.include_router(documents.router, prefix="/documents", tags=["Documents"])

@app.get("/")
async def root():
    return {"message": "Welcome to OmniRAG API"}