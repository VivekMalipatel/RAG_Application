from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import logging
from app.core.vector_store.qdrant.qdrant_handler import QdrantHandler
from app.config import settings
from app.core.embedding.embedding_handler import EmbeddingHandler
from app.core.models.model_type import ModelType

router = APIRouter()
qdrant_handler = QdrantHandler()

class SearchFilter(BaseModel):
    """Filter parameters for the search query"""
    must: Optional[Dict] = Field(None, description="Fields that must match")
    must_not: Optional[Dict] = Field(None, description="Fields that must not match")
    should: Optional[Dict] = Field(None, description="Fields that should match")
    
class SearchParams(BaseModel):
    """Dynamic search parameters for hybrid search"""
    matryoshka_64_limit: int = Field(100, description="Limit for 64-dim matryoshka embeddings")
    matryoshka_128_limit: int = Field(80, description="Limit for 128-dim matryoshka embeddings")
    matryoshka_256_limit: int = Field(60, description="Limit for 256-dim matryoshka embeddings")
    dense_limit: int = Field(40, description="Limit for dense embeddings")
    quantized_limit: int = Field(40, description="Limit for quantized embeddings")
    sparse_limit: int = Field(50, description="Limit for sparse embeddings")
    final_limit: int = Field(30, description="Final number of results to consider")
    hnsw_ef: int = Field(128, description="HNSW search parameter")
    
class SearchRequest(BaseModel):
    """Request model for hybrid search"""
    query: str = Field(..., description="Search query text")
    user_id: str = Field(..., description="User ID for collection search")
    top_k: int = Field(5, description="Number of results to return")
    search_params: Optional[SearchParams] = Field(None, description="Dynamic search parameters")
    # filters: Optional[SearchFilter] = Field(None, description="Search filters")

class SearchResponse(BaseModel):
    """Response model for search results"""
    results: List
    total_found: int
    query: str
    user_id: str

class CollectionsResponse(BaseModel):
    """Response model for collections list"""
    collections: List[str]
    total: int

class CollectionCountResponse(BaseModel):
    """Response model for collection chunk count"""
    user_id: str
    count: int
    # filters: Optional[Dict] = None

@router.post("/search", response_model=SearchResponse)
async def hybrid_search(request: SearchRequest):
    try:
        # Initialize embedding models
        embedding_handler = EmbeddingHandler(
            provider=settings.TEXT_EMBEDDING_PROVIDER,
            model_name=settings.TEXT_EMBEDDING_MODEL_NAME,
            model_type=ModelType.TEXT_EMBEDDING
        )
        
        # Generate query embeddings
        dense_vector = await embedding_handler.encode_dense(request.query)
        sparse_vector = await embedding_handler.encode_sparse(request.query)
        
        # Set default search parameters if not provided
        search_params = request.search_params
        if not search_params:
            search_params = SearchParams()
        
        # Format filters if provided
        # filters = None
        # if request.filters:
        #     filters = request.filters.model_dump(exclude_none=True)
        
        # Execute hybrid search
        search_results = await qdrant_handler.hybrid_search(
            user_id=request.user_id,
            query_text=request.query,
            dense_vector=dense_vector[0],
            sparse_vector=sparse_vector,
            top_k=request.top_k,
            search_params=search_params.model_dump(),
            # filters=filters
        )
        
        return SearchResponse(
            results=search_results,
            total_found=len(search_results),
            query=request.query,
            user_id=request.user_id
        )
    
    except Exception as e:
        logging.error(f"Error in hybrid search: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@router.get("/collections", response_model=CollectionsResponse)
async def get_all_collections():
    """
    Retrieve a list of all available collections (users).
    
    Returns:
        CollectionsResponse: The list of collections and their count
    """
    try:
        collections = await qdrant_handler.get_all_containers()
        return CollectionsResponse(
            collections=collections,
            total=len(collections)
        )
    except Exception as e:
        logging.error(f"Error fetching collections: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch collections: {str(e)}")

@router.get("/collections/{user_id}/count", response_model=CollectionCountResponse)
async def get_collection_chunk_count(
    user_id: str,
    # must: Optional[Dict] = Query(None, description="Fields that must match"),
    # must_not: Optional[Dict] = Query(None, description="Fields that must not match"),
    # should: Optional[Dict] = Query(None, description="Fields that should match")
):
    """
    Get the number of chunks in a user's collection with optional filters.
    
    Args:
        user_id (str): User ID / Collection name
        must (Dict, optional): Fields that must match
        must_not (Dict, optional): Fields that must not match
        should (Dict, optional): Fields that should match
        
    Returns:
        CollectionCountResponse: User ID and chunk count
    """
    try:
        # Format filters if provided
        filters = None
        # if must or must_not or should:
        #     filters = {}
        #     if must:
        #         filters["must"] = must
        #     if must_not:
        #         filters["must_not"] = must_not
        #     if should:
        #         filters["should"] = should
        
        count = await qdrant_handler.get_collection_chunk_count(user_id)#, filters)
        return CollectionCountResponse(
            user_id=user_id,
            count=count,
            # filters=filters
        )
    except Exception as e:
        logging.error(f"Error getting chunk count for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get chunk count: {str(e)}")