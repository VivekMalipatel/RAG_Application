import time
import uuid
from typing import List, Dict, Any, Optional, Union

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session

from db.session import get_db
from core.security import get_api_key
from db.models import ApiKey, Usage
from schemas.reranker import RerankerRequest, RerankerResponse, RerankerDocument
from schemas.chat import UsageInfo

# Import our model handlers
from model_handler import ModelRouter
from model_type import ModelType
from model_provider import Provider

router = APIRouter()

@router.post("", response_model=RerankerResponse)
async def rerank_documents(
    request: RerankerRequest,
    background_tasks: BackgroundTasks,
    api_key: ApiKey = Depends(get_api_key),
    db: Session = Depends(get_db),
):
    """
    Reranks a list of documents based on their relevance to the input query.
    
    - `query`: The search query to rank documents against
    - `documents`: Array of text strings to be reranked
    - `model`: ID of the reranker model to use
    - `max_chunks`: Maximum number of documents to return in the response
    - `return_documents`: Whether to include document text in the response
    - `user`: Optional user identifier
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    try:
        # Validate model is specified
        if not request.model:
            raise HTTPException(status_code=400, detail="Model must be specified")
            
        # Initialize model router with requested model
        model_router = await ModelRouter.initialize_from_model_name(
            model_name="jinaai/jina-colbert-v2",
            model_type=ModelType.RERANKER
        )
        
        # Rerank documents - this now returns a list of sorted indices
        sorted_indices = await model_router.rerank_documents(
            query=request.query,
            documents=request.documents,
            max_documents=request.max_chunks
        )
        
        # Prepare response data using the sorted indices
        data = []
        for rank, original_index in enumerate(sorted_indices):
            # Create reranker document with or without original text
            if request.return_documents:
                data.append(RerankerDocument(
                    document=request.documents[original_index], # Get doc using original index
                    index=original_index,
                    # relevance_score is omitted as it's not provided by the client
                ))
            else:
                data.append(RerankerDocument(
                    index=original_index
                    # relevance_score is omitted
                ))
        
        # Calculate token usage (rough estimation)
        query_chars = len(request.query)
        docs_chars = sum(len(doc) for doc in request.documents)
        total_chars = query_chars + docs_chars
        total_tokens = total_chars // 4  # Rough approximation
        
        # Log usage to database
        completion_time = time.time() - start_time
        background_tasks.add_task(
            log_usage,
            db=db,
            api_key_id=getattr(api_key, "id", None),
            request_id=request_id,
            endpoint="rerank",
            model=request.model,
            provider=model_router.provider.value,
            prompt_tokens=total_tokens,
            completion_tokens=0,  # Reranking doesn't produce completion tokens
            total_tokens=total_tokens,
            processing_time=completion_time,
            request_data=request.model_dump_json()
        )
        
        # Return response
        return RerankerResponse(
            model=request.model,
            data=data,
            order=sorted_indices, # Add the sorted indices list
            usage=UsageInfo(
                prompt_tokens=total_tokens,
                completion_tokens=0,
                total_tokens=total_tokens
            )
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
        # Use datetime object for timestamp
        import datetime
        usage_record = Usage(
            api_key_id=api_key_id,
            timestamp=datetime.datetime.utcnow(), # Use datetime object
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
        # Log the error properly
        import logging
        logging.error(f"Failed to log usage: {str(e)}")