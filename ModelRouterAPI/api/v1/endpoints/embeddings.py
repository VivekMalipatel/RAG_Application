import time
import uuid
from typing import List, Dict, Any, Optional, Union

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session

from db.session import get_db
from core.security import get_api_key
from db.models import ApiKey, Usage
from schemas.embeddings import EmbeddingRequest, EmbeddingResponse, EmbeddingData
from schemas.chat import UsageInfo

# Import our model handlers
from model_handler import ModelRouter
from model_provider import Provider
from model_type import ModelType

router = APIRouter()

@router.post("", response_model=EmbeddingResponse)
async def create_embeddings(
    request: EmbeddingRequest,
    background_tasks: BackgroundTasks,
    api_key: ApiKey = Depends(get_api_key),
    db: Session = Depends(get_db),
):
    """
    Creates embeddings for the provided input text.
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    try:
        # Convert input to list if it's a single string
        input_texts = request.input if isinstance(request.input, list) else [request.input]
        
        # Parse model info to determine the provider
        model_parts = request.model.split("/")
        if len(model_parts) > 1:
            provider = Provider.HUGGINGFACE
        elif request.model.startswith(("text-embedding", "ada")):
            provider = Provider.OPENAI
        else:
            provider = Provider.OLLAMA
        
        # Initialize the appropriate model
        model_router = ModelRouter(
            provider=provider,
            model_name=request.model,
            model_type=ModelType.TEXT_EMBEDDING,
        )
        
        # Generate embeddings
        embeddings = await model_router.embed_text(input_texts)
        
        # Estimate token usage (simpler than chat, just character count / 4)
        total_chars = sum(len(text) for text in input_texts)
        prompt_tokens = total_chars // 4  # Rough approximation
        
        # Log usage
        completion_time = time.time() - start_time
        background_tasks.add_task(
            log_usage,
            db=db,
            api_key_id=getattr(api_key, "id", None),
            request_id=request_id,
            endpoint="embeddings",
            model=request.model,
            provider=provider.value,
            prompt_tokens=prompt_tokens,
            completion_tokens=0,
            total_tokens=prompt_tokens,
            processing_time=completion_time,
            request_data=request.json()
        )
        
        # Create response
        embedding_data = []
        for i, embedding in enumerate(embeddings):
            embedding_data.append(
                EmbeddingData(
                    embedding=embedding,
                    index=i
                )
            )
        
        return EmbeddingResponse(
            data=embedding_data,
            model=request.model,
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=0,
                total_tokens=prompt_tokens
            )
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Helper function to log usage - same as in chat.py
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
    """Log API usage to database for tracking and billing."""
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
        # Log error but don't fail the request
        print(f"Error logging usage: {str(e)}")
        db.rollback()