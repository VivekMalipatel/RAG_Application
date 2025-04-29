from fastapi import APIRouter, Depends, HTTPException, Request
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

from app.services.vector_store import VectorStore
from app.core.model.model_handler import ModelHandler
import logging

logger = logging.getLogger(__name__)

router = APIRouter()
vector_store = VectorStore()
vector_store.load()
model_handler = ModelHandler()

class SearchTextRequest(BaseModel):
    text: str
    top_k: int = 10

class SearchImageRequest(BaseModel):
    image_base64: str
    text: Optional[str] = None
    top_k: int = 10

@router.post("/search/text", response_model=List[Dict[str, Any]])
async def search_by_text(request: SearchTextRequest):
    try:
        query_embeddings = model_handler.embed_text([request.text])
        
        results = vector_store.search(
            query_vectors=query_embeddings,
            k=request.top_k
        )
        
        return results
    except Exception as e:
        logger.error(f"Error during text search: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

@router.post("/search/image", response_model=List[Dict[str, Any]])
async def search_by_image(request: SearchImageRequest):
    try:
        image_text = request.text or "Image query"
        query_data = [{"image": request.image_base64, "text": image_text}]
        
        query_embeddings = model_handler.embed_image(query_data)
        
        results = vector_store.search(
            query_vectors=query_embeddings,
            k=request.top_k
        )
        
        return results
    except Exception as e:
        logger.error(f"Error during image search: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

@router.get("/stats")
async def get_vector_stats():
    return vector_store.get_stats()