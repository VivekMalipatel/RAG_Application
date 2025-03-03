from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, Any
import asyncio
from app.services.query_processor import QueryProcessor

# Initialize FastAPI Router
router = APIRouter()

# Initialize Query Processor
query_processor = QueryProcessor()

class QueryRequest(BaseModel):
    """Request schema for querying the knowledge base."""
    user_id: str
    query: str
    top_k: int = 10

@router.post("/ask", response_model=Dict[str, Any])
async def query_knowledge_base(request: QueryRequest):
    """
    Processes a user query using hybrid search and LLM.
    
    Args:
        request (QueryRequest): User query input.
    
    Returns:
        Dict[str, Any]: LLM response and supporting sources.
    """
    try:
        result = await query_processor.process_query(request.user_id, request.query, request.top_k)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")