import time
import logging
import asyncio
from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from db.session import get_db
from core.security import get_api_key
from db.models import ApiKey
from schemas.models import Model, ModelList, ModelPermission
from openai_client import OpenAIClient
from ollama.model_loader import OllamaModelLoader
from model_type import ModelType
from core.model_selector import ModelSelector

router = APIRouter()
logger = logging.getLogger(__name__)

CUSTOM_MODELS = {
    "nomic-ai/colnomic-embed-multimodal-3b": {
        "id": "nomic-ai/colnomic-embed-multimodal-3b",
        "created": int(time.time()) - 50000,
        "owned_by": "huggingface",
        "type": "embedding"
    },
    "jinaai/jina-colbert-v2": {
        "id": "jinaai/jina-colbert-v2", 
        "created": int(time.time()) - 45000,
        "owned_by": "huggingface",
        "type": "reranker"
    }
}

async def fetch_openai_models() -> Dict[str, Dict[str, Any]]:
    models = {}
    try:
        client = OpenAIClient()
        all_models = await client.get_model_list()
        
        text_gen_patterns = ["gpt-", "text-davinci", "davinci", "claude", "o1-mini", "o1-preview", "o2", "o3", "o4", "deepseek"]
        
        text_gen_models = [
            model_id for model_id in all_models 
            if any(pattern in model_id.lower() for pattern in text_gen_patterns)
        ]
        
        for model_id in text_gen_models:
            models[model_id] = {
                "id": model_id,
                "created": int(time.time()) - 10000,
                "owned_by": "openai",
                "type": "completion"
            }
        logger.info(f"Fetched {len(models)} text generation models from OpenAI")
    except Exception as e:
        logger.error(f"Error fetching OpenAI models: {str(e)}")
    
    return models

async def fetch_ollama_models() -> Dict[str, Dict[str, Any]]:
    models = {}
    try:
        ollama_loader = OllamaModelLoader()
        model_list = await ollama_loader.get_available_models()
        
        for model_data in model_list:
            model_id = model_data["name"].split(":")[0]
            models[model_id] = {
                "id": model_id,
                "created": int(time.time()) - 30000,
                "owned_by": "ollama",
                "type": "completion"
            }
        logger.info(f"Fetched {len(models)} models from Ollama")
    except Exception as e:
        logger.error(f"Error fetching Ollama models: {str(e)}")
    
    return models

@router.get("", response_model=ModelList)
async def list_models(
    api_key: ApiKey = Depends(get_api_key),
    db: Session = Depends(get_db),
):
    combined_models = CUSTOM_MODELS.copy()
    
    openai_task = asyncio.create_task(fetch_openai_models())
    ollama_task = asyncio.create_task(fetch_ollama_models())
    
    openai_models = await openai_task
    ollama_models = await ollama_task
    
    combined_models.update(openai_models)
    combined_models.update(ollama_models)
    
    models = []
    for model_id, model_data in combined_models.items():
        default_permission = ModelPermission(
            id=f"modelperm-{model_id}",
            created=model_data["created"],
            allow_create_engine=False,
            allow_sampling=True,
            allow_logprobs=True,
            allow_search_indices=False,
            allow_view=True,
            allow_fine_tuning=False,
            organization="*",
            is_blocking=False,
        )
        
        model = Model(
            id=model_id,
            created=model_data["created"],
            owned_by=model_data["owned_by"],
            permission=[default_permission],
        )
        models.append(model)
    
    return ModelList(data=models)


@router.get("/{model_id}", response_model=Model)
async def get_model(
    model_id: str,
    api_key: ApiKey = Depends(get_api_key),
    db: Session = Depends(get_db),
):
    if model_id in CUSTOM_MODELS:
        model_data = CUSTOM_MODELS[model_id]
    else:
        model_selector = ModelSelector()
        try:
            provider, actual_model_id = await model_selector.select_best_model(
                model_type=ModelType.TEXT_GENERATION,
                model_name=model_id
            )
            
            if actual_model_id != model_id:
                model_id = actual_model_id
                
            if provider == "openai":
                model_data = {
                    "id": model_id,
                    "created": int(time.time()) - 10000,
                    "owned_by": "openai"
                }
            elif provider == "ollama":
                model_data = {
                    "id": model_id,
                    "created": int(time.time()) - 30000,
                    "owned_by": "ollama"
                }
            else:
                model_data = {
                    "id": model_id,
                    "created": int(time.time()) - 40000,
                    "owned_by": "huggingface"
                }
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found: {str(e)}")
    
    default_permission = ModelPermission(
        id=f"modelperm-{model_id}",
        created=model_data["created"],
        allow_create_engine=False,
        allow_sampling=True,
        allow_logprobs=True,
        allow_search_indices=False,
        allow_view=True,
        allow_fine_tuning=False,
        organization="*",
        is_blocking=False,
    )
    
    model = Model(
        id=model_id,
        created=model_data["created"],
        owned_by=model_data["owned_by"],
        permission=[default_permission],
    )
    
    return model