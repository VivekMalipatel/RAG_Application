import time
import uuid
from typing import Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from db.session import get_db
from core.security import get_api_key
from db.models import ApiKey, Usage

from model_handler_v2 import ModelRouterV2
from model_type import ModelType

router = APIRouter()

def create_error_response(message: str, code: str, status_code: int = 400):
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "param": None,
                "code": code
            }
        }
    )

@router.post("")
async def create_embeddings(
    request: Request,
    background_tasks: BackgroundTasks,
    api_key: ApiKey = Depends(get_api_key),
    db: Session = Depends(get_db),
):
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    try:
        request_data = await request.json()
        
        if not request_data.get("model"):
            return create_error_response(
                "Model parameter is required",
                "invalid_request_error", 
                400
            )
        
        input_data = request_data.get("input")
        if isinstance(input_data, list):
            is_image_input = all(isinstance(item, dict) for item in input_data) if input_data else False
        else:
            is_image_input = isinstance(input_data, dict)
        
        try:
            if is_image_input:
                model_router = await ModelRouterV2.initialize_from_model_name(
                    model_name=request_data.get("model"),
                    model_type=ModelType.IMAGE_EMBEDDING
                )
            else:
                model_router = await ModelRouterV2.initialize_from_model_name(
                    model_name=request_data.get("model"),
                    model_type=ModelType.TEXT_EMBEDDING
                )
        except Exception as model_error:
            return create_error_response(
                f"The model '{request_data.get('model')}' does not exist or is not available",
                "model_not_found",
                404
            )
        
        try:
            response = await model_router.embed(**request_data)
            
            completion_time = time.time() - start_time
            background_tasks.add_task(
                log_usage,
                db=db,
                api_key_id=getattr(api_key, "id", None),
                request_id=request_id,
                endpoint="embeddings",
                model=request_data.get("model"),
                provider=model_router.provider.value,
                prompt_tokens=response.get("usage", {}).get("prompt_tokens", 0),
                completion_tokens=0,
                total_tokens=response.get("usage", {}).get("total_tokens", 0),
                processing_time=completion_time,
                request_data=str(request_data)
            )
            
            return response
            
        except Exception as e:
            return create_error_response(
                f"Error generating embeddings: {str(e)}",
                "embedding_error",
                500
            )
    
    except Exception as e:
        return create_error_response(
            str(e),
            "internal_server_error",
            500
        )

def log_usage(
    db: Session, 
    api_key_id: Optional[int],
    request_id: str,
    endpoint: str,
    model: str,
    provider: str,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    processing_time: float,
    request_data: str
):
    try:
        import datetime
        
        usage = Usage(
            request_id=request_id,
            api_key_id=api_key_id,
            endpoint=endpoint,
            model=model,
            provider=provider,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            processing_time=processing_time,
            request_data=request_data,
            timestamp=datetime.datetime.utcnow()
        )
        db.add(usage)
        db.commit()
    except Exception as e:
        import logging
        logging.error(f"Failed to log usage: {str(e)}")
