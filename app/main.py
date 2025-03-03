from fastapi import FastAPI
from app.api.v1.endpoints import agent, documents, user, upload, minio_webhook, query
from app.api.db.base import init_db, close_db
from app.core.storage_bin.minio.session_minio import minio_session
from app.core.cache.session_redis import redis_session
from app.core.vector_store.qdrant.qdrant_session import qdrant_session
from app.core.storage_bin.minio.setup_minio import MinIOSetup 
import uvicorn
import asyncio
import logging
from app.services.file_processor.file_processor import FileEventProcessor
from app.config import settings



async def lifespan(app: FastAPI):
    """Application lifecycle management"""
    try:
        # Initialize core services
        await init_db()
        await minio_session.connect()
        await redis_session.connect()
        await qdrant_session.connect()
        minio_setup = MinIOSetup()
        asyncio.create_task(minio_setup.setup_minio())

        # Start file processors
        processor = FileEventProcessor()
        asyncio.create_task(processor.process_events())

        logging.info("All services initialized successfully")
    except Exception as e:
        logging.critical(f"Failed to initialize application: {str(e)}")
        raise

    yield  # App is running

    # Cleanup
    await close_db()
    await minio_session.close()
    await redis_session.close()
    await qdrant_session.close()
    logging.info("Services shut down successfully")

app = FastAPI(
    title="OmniRAG API",
    description="API for OmniRAG's document processing and retrieval",
    version="1.0.0",
    lifespan=lifespan
)

# Mount routers with correct prefix
app.include_router(user.router, prefix="/api/v1/users", tags=["Users"])
app.include_router(agent.router, prefix="/api/v1/agents", tags=["Agents"])
app.include_router(documents.router, prefix="/api/v1/documents", tags=["Documents"])
app.include_router(upload.router, prefix="/api/v1/files", tags=["File Upload"])
app.include_router(minio_webhook.router, prefix="/api/v1/minio", tags=["MinIO Webhooks"])
app.include_router(query.router, prefix="/api/v1/query", tags=["Query"])

@app.get("/")
async def root():
    return {"message": "Welcome to OmniRAG API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=settings.DEBUG_MODE,
        log_level=settings.LOG_LEVEL
    )
