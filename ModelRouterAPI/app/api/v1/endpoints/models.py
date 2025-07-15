import time
import logging
from fastapi import APIRouter
from config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("", response_model=None)
async def list_models():
    models = []
    
    all_models = (
        settings.TEXT_GENERATION_MODELS +
        settings.TEXT_EMBEDDING_MODELS +
        settings.IMAGE_EMBEDDING_MODELS +
        settings.RERANKER_MODELS
    )
    
    for model_id in all_models:
        provider_config = settings.get_provider_config(model_id)
        if provider_config:
            owned_by = provider_config['provider_name'].lower()
        else:
            owned_by = "openai"
        
        model = {
            "id": model_id,
            "object": "model",
            "created": int(time.time()) - 10000,
            "owned_by": owned_by,
        }
        models.append(model)
    
    return {"object": "list", "data": models}

@router.get("/{model_id}", response_model=None)
async def get_model(
    model_id: str,
):
    all_models = (
        settings.TEXT_GENERATION_MODELS +
        settings.TEXT_EMBEDDING_MODELS +
        settings.IMAGE_EMBEDDING_MODELS +
        settings.RERANKER_MODELS
    )
    
    if model_id not in all_models:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    
    provider_config = settings.get_provider_config(model_id)
    if provider_config:
        owned_by = provider_config['provider_name'].lower()
    else:
        owned_by = "openai"
    
    model = {
        "id": model_id,
        "object": "model",
        "created": int(time.time()) - 10000,
        "owned_by": owned_by,
    }
    
    return model
