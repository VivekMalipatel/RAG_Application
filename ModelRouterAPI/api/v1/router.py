from fastapi import APIRouter

from api.v1.endpoints import chat, embeddings, models, ollama_loader, reranker, structured

api_router = APIRouter()

# Updated paths to match OpenAI API structure
api_router.include_router(models.router, prefix="/models", tags=["Models"])
api_router.include_router(chat.router, tags=["Chat"])  # Removed prefix to match OpenAI paths
api_router.include_router(embeddings.router, prefix="/embeddings", tags=["Embeddings"])
api_router.include_router(reranker.router, prefix="/rerank", tags=["Reranking"])
api_router.include_router(ollama_loader.router, prefix="/ollama", tags=["Ollama"])
api_router.include_router(structured.router, prefix="/structured", tags=["Structured Output"])