from fastapi import APIRouter

from api.v1.endpoints import chat, embeddings, models, ollama_loader, reranker

api_router = APIRouter()

api_router.include_router(models.router, prefix="/models", tags=["Models"])
api_router.include_router(chat.router, tags=["Chat"])
api_router.include_router(embeddings.router, prefix="/embeddings", tags=["Embeddings"])
api_router.include_router(reranker.router, prefix="/rerank", tags=["Reranking"])
api_router.include_router(ollama_loader.router, prefix="/ollama", tags=["Ollama"])