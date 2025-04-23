import time
import uuid
from typing import List, Dict, Any, Optional, Union
import numpy as np

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session

from db.session import get_db
from core.security import get_api_key
from db.models import ApiKey, Usage
from schemas.reranker import RerankerRequest, RerankerResponse, RerankerScoreItem
from schemas.chat import UsageInfo

# Import our model handlers
from model_handler import ModelRouter
from model_type import ModelType

router = APIRouter()

# Hardcoded model for reranking
RERANKER_MODEL = "jinaai/jina-colbert-v2"

@router.post("", response_model=RerankerResponse)
async def rerank_documents(
    request: RerankerRequest,
    background_tasks: BackgroundTasks,
    api_key: ApiKey = Depends(get_api_key),
    db: Session = Depends(get_db),
):
    """
    Reranks a list of documents based on relevance to a query.
    Uses a hardcoded Hugging Face reranker model.
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    try:
        # Use hardcoded model name instead of the one from the request
        model_router = ModelRouter.initialize_from_model_name(
            model_name=RERANKER_MODEL,
            model_type=ModelType.RERANKER,
        )
        
        # Rerank documents
        ranked_indices = await model_router.rerank_documents(
            query=request.query, 
            documents=request.documents, 
            max_tokens=request.max_tokens
        )
        
        # Generate scores (normalize to 0-1 range)
        # This is a simple approximation since we don't have actual scores from the ranking
        scores = np.linspace(1.0, 0.1, len(ranked_indices))
        
        # Create results with document indices and scores
        results = []
        for i, idx in enumerate(ranked_indices):
            results.append(
                RerankerScoreItem(
                    document_index=idx,
                    score=float(scores[i])
                )
            )
        
        # Estimate token usage (rough approximation)
        query_chars = len(request.query)
        docs_chars = sum(len(doc) for doc in request.documents)
        total_chars = query_chars + docs_chars
        # Very rough token estimation (4 chars per token is an approximation)
        total_tokens = total_chars // 4
        
        # Log usage
        completion_time = time.time() - start_time
        background_tasks.add_task(
            log_usage,
            db=db,
            api_key_id=getattr(api_key, "id", None),
            request_id=request_id,
            endpoint="reranker",
            model=RERANKER_MODEL,
            provider="huggingface",  # We know it's always huggingface
            prompt_tokens=total_tokens,
            completion_tokens=0,
            total_tokens=total_tokens,
            processing_time=completion_time,
            request_data=request.json()
        )
        
        # Create response
        return RerankerResponse(
            model=RERANKER_MODEL,  # Return the actual model used, not the requested one
            results=results,
            usage=UsageInfo(
                prompt_tokens=total_tokens,
                completion_tokens=0,
                total_tokens=total_tokens
            )
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Helper function to log usage - same as in other endpoint files
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