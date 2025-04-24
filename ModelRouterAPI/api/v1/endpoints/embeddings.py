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

# Import our model handlers
from model_handler import ModelRouter
from model_type import ModelType
from model_provider import Provider

router = APIRouter()

# OpenAI-style error response
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
    """
    Creates an embedding vector representing the input text.
    
    - `input`: Input text to embed, can be a string or array of strings
    - `model`: ID of the embedding model to use
    - `dimensions`: Optional number of dimensions for the output embeddings
    - `encoding_format`: Format to return embeddings in ("float" or "base64")
    - `user`: Optional user identifier
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    try:
        # Convert input to list if it's a single string
        input_texts = request.input if isinstance(request.input, list) else [request.input]
        
        # OpenAI-style validation: Check for empty inputs
        if any(not text.strip() for text in input_texts):
            return create_error_response(
                "One or more input strings are empty. Input strings must be non-empty.",
                "invalid_input_empty",
                400
            )
            
        # Validate model is specified
        if not request.model:
            return create_error_response(
                "Model parameter is required",
                "invalid_request_error", 
                400
            )
        
        # Validate encoding format
        if request.encoding_format not in ["float", "base64"]:
            return create_error_response(
                "Encoding format must be either 'float' or 'base64'",
                "invalid_encoding_format",
                400
            )
            
        # Initialize model router with requested model
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
        
        # Generate embeddings
        embeddings = await model_router.embed_text(texts=input_texts)
        
        # Convert to base64 if requested
        if request.encoding_format == "base64":
            # Handle nested structure if present (Nomic multimodal models)
            if embeddings and isinstance(embeddings, list) and embeddings and isinstance(embeddings[0], list) and \
               embeddings[0] and isinstance(embeddings[0][0], list):
                # Triple nested list - use numpy's recursive conversion
                embeddings = [
                    [base64.b64encode(np.array(emb_set, dtype=np.float32).tobytes()).decode('utf-8') 
                     for emb_set in embedding]
                    for embedding in embeddings
                ]
            else:
                # Standard flat embeddings
                embeddings = [
                    base64.b64encode(np.array(embedding, dtype=np.float32).tobytes()).decode('utf-8')
                    for embedding in embeddings
                ]
        
        # Estimate token usage (simple estimation)
        prompt_tokens = sum(len(text.split()) for text in input_texts)
        
        # Prepare response
        embedding_data = [
            EmbeddingData(embedding=embedding, index=i, object="embedding")
            for i, embedding in enumerate(embeddings)
        ]
        
        # Log usage to database
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
            completion_tokens=0,  # Embeddings don't have completion tokens
            total_tokens=prompt_tokens,
            processing_time=completion_time,
            request_data=request.model_dump_json()
        )
        
        # Return OpenAI-compatible response
        return EmbeddingResponse(
            data=embedding_data,
            model=request.model,
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=0,
                total_tokens=prompt_tokens
            ),
            object="list"  # Standard OpenAI field
        )
    
    except Exception as e:
        # Return error in OpenAI format
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
    """Log API usage to the database"""
    try:
        # Convert Unix timestamp to datetime object
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
            # Use datetime object instead of integer timestamp
            timestamp=datetime.datetime.utcnow()
        )
        db.add(usage)
        db.commit()
    except Exception as e:
        import logging
        logging.error(f"Failed to log usage: {str(e)}")