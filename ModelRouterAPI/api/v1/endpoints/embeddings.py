import time
import uuid
import base64
from typing import List, Dict, Any, Optional, Union

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
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

@router.post("", response_model=EmbeddingResponse)
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
        
        # Validate model is specified
        if not request.model:
            raise HTTPException(status_code=400, detail="Model must be specified")
            
        # Initialize model router with requested model
        model_router = await ModelRouter.initialize_from_model_name(
            model_name="nomic-ai/colnomic-embed-multimodal-7b",
            model_type=ModelType.TEXT_EMBEDDING
        )
        
        # Set up extra parameters for embedding
        model_kwargs = {}
        if request.dimensions:
            model_kwargs["dimensions"] = request.dimensions
            
        # Generate embeddings with optional dimension parameter
        embeddings = await model_router.client.embed_text(
            texts=input_texts,
            dimensions=request.dimensions
        )
        
        # Convert to base64 if requested
        if request.encoding_format == "base64":
            embeddings = [
                base64.b64encode(np.array(embedding, dtype=np.float32).tobytes()).decode('utf-8')
                for embedding in embeddings
            ]
            
        # Calculate token usage
        total_chars = sum(len(text) for text in input_texts)
        prompt_tokens = total_chars // 4  # Rough estimation
        
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
        raise HTTPException(status_code=500, detail=str(e))

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
    """Log API usage to database"""
    try:
        usage_record = Usage(
            api_key_id=api_key_id,
            timestamp=time.time(),
            endpoint=endpoint,
            model=model,
            provider=provider,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            processing_time=processing_time,
            request_id=request_id,
            request_data=request_data
        )
        db.add(usage_record)
        db.commit()
    except Exception as e:
        db.rollback()