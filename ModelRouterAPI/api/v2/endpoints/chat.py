import time
import uuid
from typing import Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Request
from fastapi.responses import StreamingResponse, JSONResponse
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

@router.post("/chat/completions", response_model=None)
async def create_chat_completion(
    request: Request,
    background_tasks: BackgroundTasks,
    api_key: ApiKey = Depends(get_api_key),
    db: Session = Depends(get_db)
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
        
        try:
            model_router = await ModelRouterV2.initialize_from_model_name(
                model_name=request_data.get("model"),
                model_type=ModelType.TEXT_GENERATION
            )
        except Exception as model_error:
            return create_error_response(
                f"The model '{request_data.get('model')}' does not exist or is not available",
                "model_not_found",
                404
            )
        
        try:
            if request_data.get("stream", False):
                return StreamingResponse(
                    generate_chat_stream(request_data, model_router, background_tasks, api_key, db, start_time, request_id),
                    media_type="text/event-stream"
                )
            else:
                response = await model_router.generate_text(**request_data)
                
                completion_time = time.time() - start_time
                background_tasks.add_task(
                    log_usage,
                    db=db,
                    api_key_id=getattr(api_key, "id", None),
                    request_id=request_id,
                    endpoint="chat/completions",
                    model=request_data.get("model"),
                    provider=model_router.provider.value,
                    prompt_tokens=response.get("usage", {}).get("prompt_tokens", 0),
                    completion_tokens=response.get("usage", {}).get("completion_tokens", 0),
                    total_tokens=response.get("usage", {}).get("total_tokens", 0),
                    processing_time=completion_time,
                    request_data=str(request_data)
                )
                
                return response
                
        except Exception as e:
            return create_error_response(
                f"Error generating chat completion: {str(e)}",
                "chat_completion_error",
                500
            )
    
    except Exception as e:
        return create_error_response(
            str(e),
            "internal_server_error",
            500
        )

async def generate_chat_stream(
    request_data: Dict[str, Any],
    model_router: ModelRouterV2,
    background_tasks: BackgroundTasks,
    api_key: ApiKey,
    db: Session,
    start_time: float,
    request_id: str
):
    try:
        stream = await model_router.generate_text(**request_data)
        
        accumulated_usage = {}
        async for chunk in stream:
            if isinstance(chunk, dict) and chunk.get("usage"):
                accumulated_usage = chunk["usage"]
            
            if isinstance(chunk, str):
                yield f"data: {chunk}\n\n"
            else:
                import json
                yield f"data: {json.dumps(chunk)}\n\n"
        
        yield "data: [DONE]\n\n"
        
        completion_time = time.time() - start_time
        background_tasks.add_task(
            log_usage,
            db=db,
            api_key_id=getattr(api_key, "id", None),
            request_id=request_id,
            endpoint="chat/completions",
            model=request_data.get("model"),
            provider=model_router.provider.value,
            prompt_tokens=accumulated_usage.get("prompt_tokens", 0) if accumulated_usage else 0,
            completion_tokens=accumulated_usage.get("completion_tokens", 0) if accumulated_usage else 0,
            total_tokens=accumulated_usage.get("total_tokens", 0) if accumulated_usage else 0,
            processing_time=completion_time,
            request_data=str(request_data)
        )
        
    except Exception as e:
        import json
        error_chunk = {
            "error": {
                "message": f"Error in streaming: {str(e)}",
                "type": "streaming_error"
            }
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
        yield "data: [DONE]\n\n"

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
