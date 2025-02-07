from fastapi import FastAPI
from app.api.v1.endpoints import agent, documents, user
from app.api.db.base import init_db, close_db
from app.core.storage_bin.minio import minio_session
from app.core.cache import redis_session
import uvicorn


async def lifespan(app: FastAPI):
    """Startup and Shutdown events for FastAPI."""
    
    await init_db()
    await minio_session.connect()
    await redis_session.connect()

    yield

    await close_db() 
    await minio_session.close()
    await redis_session.close()

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

if __name__ == "__main__":

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)