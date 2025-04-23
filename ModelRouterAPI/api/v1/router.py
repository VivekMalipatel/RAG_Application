from fastapi import APIRouter

from api.v1.endpoints import chat, completions, embeddings, models

api_router = APIRouter()

api_router.include_router(models.router, tags=["Models"])
api_router.include_router(completions.router, tags=["Completions"])
api_router.include_router(chat.router, tags=["Chat"])
api_router.include_router(embeddings.router, tags=["Embeddings"])