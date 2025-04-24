from pydantic import BaseModel, Field
from typing import List, Optional, Union, Dict, Any

from schemas.chat import UsageInfo

class RerankerRequest(BaseModel):
    """Request schema for document reranking"""
    model: str = Field(..., description="ID of the reranker model to use")
    query: str = Field(..., description="Query text to rank documents against")
    documents: List[str] = Field(..., description="List of document texts to rerank")
    max_chunks: Optional[int] = Field(None, description="Maximum number of chunks to return in the response")
    return_documents: Optional[bool] = Field(True, description="Whether to include the document text in the response")
    user: Optional[str] = Field(None, description="A unique identifier representing your end-user")
    
    # Hidden field for extra parameters not in the API but useful for implementation
    model_extra: Dict[str, Any] = Field(default_factory=dict, exclude=True)

class RerankerDocument(BaseModel):
    """Individual document with reranking score"""
    object: str = "reranked_document"
    document: Optional[str] = None
    index: int = Field(..., description="Original index of the document in the input array")
    relevance_score: float = Field(..., description="Relevance score (higher is better)")
    
class RerankerResponse(BaseModel):
    """Response schema for document reranking"""
    object: str = "rerank-result"
    model: str
    data: List[RerankerDocument] = Field(..., description="Reranked documents with scores")
    usage: UsageInfo