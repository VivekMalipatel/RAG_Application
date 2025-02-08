from fastapi import FastAPI
from app.api.v1.endpoints import agent, documents, user, upload
from app.api.db.base import init_db, close_db
from app.core.storage_bin.minio.session_minio import minio_session
from app.core.cache.session_redis import redis_session
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

async def lifespan(app: FastAPI):
    """Startup and Shutdown events for FastAPI."""
    await init_db()  
    await minio_session.connect()  
    await redis_session.connect()
    yield
    await close_db()
    await minio_session.close()
    await redis_session.close()

app = FastAPI(
    title="OmniRAG API",
    description="API for OmniRAG's document processing and retrieval",
    version="1.0.0",
    lifespan=lifespan
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Mount routers with correct prefix
app.include_router(user.router, prefix="/api/v1/users", tags=["Users"])
app.include_router(agent.router, prefix="/api/v1/agents", tags=["Agents"])
app.include_router(documents.router, prefix="/api/v1/documents", tags=["Documents"])
app.include_router(upload.router, prefix="/api/v1/files", tags=["File Upload"])

@app.get("/")
async def root():
    return {"message": "Welcome to OmniRAG API"}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True, log_level="debug")