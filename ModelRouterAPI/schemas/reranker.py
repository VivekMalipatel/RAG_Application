from pydantic import BaseModel, Field
from typing import List, Optional, Union, Dict

from schemas.chat import UsageInfo

class RerankerRequest(BaseModel):
    """Request schema for document reranking."""
    model: Optional[str] = None  # Model is now optional since we use a hardcoded one
    query: str = Field(..., description="Query text to rank documents against")
    documents: List[str] = Field(..., description="List of document texts to rerank")
    max_tokens: Optional[int] = Field(512, description="Maximum tokens to consider per document")
    user: Optional[str] = None

class RerankerScoreItem(BaseModel):
    """Individual document score from reranking."""
    document_index: int = Field(..., description="Original index of the document")
    score: float = Field(..., description="Relevance score (higher is better)")
    
class RerankerResponse(BaseModel):
    """Response schema for document reranking."""
    object: str = "rerank-result"
    model: str
    results: List[RerankerScoreItem] = Field(..., description="Reranked document indices with scores")
    usage: UsageInfo