from fastapi import APIRouter

from api.v1.endpoints import chat, embeddings, models

api_router = APIRouter()

api_router.include_router(models.router, prefix="/models", tags=["Models"])
api_router.include_router(chat.router, tags=["Chat"])
api_router.include_router(embeddings.router, prefix="/embeddings", tags=["Embeddings"])
