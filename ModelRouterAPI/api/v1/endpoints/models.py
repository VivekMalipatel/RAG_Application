import time
from typing import List
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from db.session import get_db
from core.security import get_api_key
from db.models import ApiKey
from schemas.models import Model, ModelList, ModelPermission

router = APIRouter()

# Hardcoded model definitions - in production, these would come from a database
AVAILABLE_MODELS = {
    # OpenAI models
    "gpt-4": {
        "id": "gpt-4",
        "created": int(time.time()) - 10000,
        "owned_by": "openai",
    },
    "gpt-3.5-turbo": {
        "id": "gpt-3.5-turbo",
        "created": int(time.time()) - 20000,
        "owned_by": "openai",
    },
    # Ollama models
    "llama2": {
        "id": "llama2",
        "created": int(time.time()) - 30000,
        "owned_by": "ollama",
    },
    "mistral": {
        "id": "mistral",
        "created": int(time.time()) - 35000,
        "owned_by": "ollama",
    },
    # Hugging Face models
    "mistralai/Mistral-7B-Instruct-v0.2": {
        "id": "mistralai/Mistral-7B-Instruct-v0.2",
        "created": int(time.time()) - 40000,
        "owned_by": "huggingface",
    },
}

@router.get("", response_model=ModelList)
async def list_models(
    api_key: ApiKey = Depends(get_api_key),
    db: Session = Depends(get_db),
):
    """
    Lists the currently available models, and provides basic information about each one.
    """
    models = []
    
    for model_id, model_data in AVAILABLE_MODELS.items():
        # Create default permissions
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
    """
    Retrieves a model instance, providing basic information about it such as the owner and permissions.
    """
    if model_id not in AVAILABLE_MODELS:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    
    model_data = AVAILABLE_MODELS[model_id]
    
    # Create default permissions
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