import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query
from pydantic import BaseModel

from db.session import get_db
from core.security import get_api_key
from ollama.model_loader import OllamaModelLoader
from sqlalchemy.orm import Session

router = APIRouter()
logger = logging.getLogger(__name__)

class ModelLoadRequest(BaseModel):
    hf_repo: str
    quantization: str = "Q8_0"
    force_convert: bool = False
    model_tag: Optional[str] = None
    enable_adapter: bool = False

class ModelLoadResponse(BaseModel):
    success: bool
    model_name: Optional[str] = None
    message: str
    task_id: Optional[str] = None

class TaskStatusResponse(BaseModel):
    task_id: str
    status: str

@router.post("/load", response_model=ModelLoadResponse)
async def load_model_from_huggingface(
    request: ModelLoadRequest,
    background_tasks: BackgroundTasks,
    api_key = Depends(get_api_key),
    db: Session = Depends(get_db),
):
    try:
        model_loader = OllamaModelLoader()
        
        task_id = f"{request.hf_repo.replace('/', '_')}_{request.quantization}"
        
        background_tasks.add_task(
            model_loader.ensure_model_available,
            hf_repo=request.hf_repo,
            quantization=request.quantization,
            force_convert=request.force_convert
        )
        
        return ModelLoadResponse(
            success=True,
            message=f"Model loading process started for {request.hf_repo}. You can check the status using the task ID.",
            task_id=task_id
        )
            
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{task_id}", response_model=TaskStatusResponse)
async def check_task_status(
    task_id: str,
    api_key = Depends(get_api_key),
    db: Session = Depends(get_db),
):
    try:
        model_loader = OllamaModelLoader()
        
        status = await model_loader.get_task_status(task_id)
        
        return TaskStatusResponse(
            task_id=task_id,
            status=status
        )
    except Exception as e:
        logger.error(f"Error checking task status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/available", response_model=List[Dict[str, Any]])
async def get_available_ollama_models(
    api_key = Depends(get_api_key),
    db: Session = Depends(get_db),
):
    try:
        model_loader = OllamaModelLoader()
        
        models = await model_loader.get_available_models()
        
        return models
            
    except Exception as e:
        logger.error(f"Error getting available models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/check/{model_name}")
async def check_model_availability(
    model_name: str,
    api_key = Depends(get_api_key),
    db: Session = Depends(get_db),
):
    try:
        model_loader = OllamaModelLoader()
        
        is_available = await model_loader.is_model_available(model_name)
        
        return {
            "model_name": model_name,
            "available": is_available
        }
            
    except Exception as e:
        logger.error(f"Error checking model availability: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))