import time
import uuid
import base64
from typing import List, Dict, Any, Optional, Union

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
import numpy as np

from db.session import get_db
from core.security import get_api_key
from db.models import ApiKey, Usage
from schemas.embeddings import EmbeddingRequest, EmbeddingResponse, EmbeddingData
from schemas.chat import UsageInfo

from model_handler import ModelRouter
from model_type import ModelType
from model_provider import Provider

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
    request: EmbeddingRequest,
    background_tasks: BackgroundTasks,
    api_key: ApiKey = Depends(get_api_key),
    db: Session = Depends(get_db),
):
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    try:
        input_texts = request.input if isinstance(request.input, list) else [request.input]
            
        if not request.model:
            return create_error_response(
                "Model parameter is required",
                "invalid_request_error", 
                400
            )
        
        if request.encoding_format not in ["float", "base64"]:
            return create_error_response(
                "Encoding format must be either 'float' or 'base64'",
                "invalid_encoding_format",
                400
            )

        is_image_input = all(isinstance(item, dict) for item in input_texts) if input_texts else False
        
        try:
            if is_image_input:
                #TODO: Add image validation logic here
                try:
                    model_router = await ModelRouter.initialize_from_model_name(
                        model_name=request.model,
                        model_type=ModelType.IMAGE_EMBEDDING
                    )
                except Exception as model_error:
                    return create_error_response(
                        f"The model '{request.model}' does not exist or is not available",
                        "model_not_found",
                        404
                    )
                embeddings = await model_router.embed_image(image=input_texts)
            else:
                try:
                    model_router = await ModelRouter.initialize_from_model_name(
                        model_name=request.model,
                        model_type=ModelType.TEXT_EMBEDDING
                    )
                except Exception as model_error:
                    return create_error_response(
                        f"The model '{request.model}' does not exist or is not available",
                        "model_not_found",
                        404
                    )
                embeddings = await model_router.embed_text(texts=input_texts)
        except Exception as e:
            return create_error_response(
                f"Error generating embeddings: {str(e)}",
                "embedding_error",
                500
            )
        
        if request.encoding_format == "base64":
            if embeddings and isinstance(embeddings, list) and embeddings and isinstance(embeddings[0], list) and \
               embeddings[0] and isinstance(embeddings[0][0], list):
                embeddings = [
                    [base64.b64encode(np.array(emb_set, dtype=np.float32).tobytes()).decode('utf-8') 
                     for emb_set in embedding]
                    for embedding in embeddings
                ]
            else:
                embeddings = [
                    base64.b64encode(np.array(embedding, dtype=np.float32).tobytes()).decode('utf-8')
                    for embedding in embeddings
                ]
        
        if is_image_input:
            prompt_tokens = len(input_texts) * 100 
        else:
            prompt_tokens = sum(len(text.split()) for text in input_texts)
        
        embedding_data = [
            EmbeddingData(embedding=embedding, index=i, object="embedding")
            for i, embedding in enumerate(embeddings)
        ]
        
        completion_time = time.time() - start_time
        background_tasks.add_task(
            log_usage,
            db=db,
            api_key_id=getattr(api_key, "id", None),
            request_id=request_id,
            endpoint="embeddings",
            model=request.model,
            provider=model_router.provider.value,
            prompt_tokens=prompt_tokens,
            completion_tokens=0,
            total_tokens=prompt_tokens,
            processing_time=completion_time,
            request_data=request.model_dump_json()
        )
        
        return EmbeddingResponse(
            data=embedding_data,
            model=request.model,
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=0,
                total_tokens=prompt_tokens
            ),
            object="list"
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