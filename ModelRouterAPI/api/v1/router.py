from fastapi import APIRouter

from api.v1.endpoints import chat, completions, embeddings, models, ollama_loader, reranker, structured

api_router = APIRouter()

api_router.include_router(models.router, prefix="/models", tags=["Models"])
api_router.include_router(completions.router, prefix="/completions", tags=["Completions"])
api_router.include_router(chat.router, prefix="/chat", tags=["Chat"])
api_router.include_router(embeddings.router, prefix="/embeddings", tags=["Embeddings"])
api_router.include_router(reranker.router, prefix="/rerank", tags=["Reranking"])
api_router.include_router(ollama_loader.router, prefix="/ollama", tags=["Ollama"])
api_router.include_router(structured.router, prefix="/structured", tags=["Structured Output"])