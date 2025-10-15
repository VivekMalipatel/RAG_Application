import asyncio
import logging
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from core.storage.neo4j_handler import get_neo4j_handler
from core.model.model_handler import get_global_model_handler

router = APIRouter()
logger = logging.getLogger(__name__)

class CypherQueryRequest(BaseModel):
    query: str
    parameters: Optional[Dict[str, Any]] = None

class SearchScoutRequest(BaseModel):
    query: str
    top_k: int = 5
    user_id: str
    org_id: str

class SearchScoutResult(BaseModel):
    space: str
    score: float
    document_id: Optional[str] = None
    page_number: Optional[int] = None
    sheet_name: Optional[str] = None
    filename: Optional[str] = None
    snippet: Optional[str] = None
    summary: Optional[str] = None
    entity_id: Optional[str] = None
    entity_type: Optional[str] = None
    entity_profile: Optional[str] = None
    column_name: Optional[str] = None
    column_profile: Optional[str] = None
    relation_type: Optional[str] = None
    relation_profile: Optional[str] = None
    source_entity_id: Optional[str] = None
    target_entity_id: Optional[str] = None
    user_id: Optional[str] = None
    org_id: Optional[str] = None

@router.post("/cypher", response_model=List[Dict[str, Any]])
async def execute_cypher_query(request: CypherQueryRequest):
    try:
        neo4j_handler = get_neo4j_handler()
        result = await neo4j_handler.execute_cypher_query(query=request.query, parameters=request.parameters)
        async def remove_embeddings_from_record(record):
            for key, value in record.items():
                if isinstance(value, dict):
                    keys_to_remove = [k for k in value.keys() if "embedding" in k.lower()]
                    for key_to_remove in keys_to_remove:
                        value.pop(key_to_remove, None)
            return record
        await asyncio.gather(*[remove_embeddings_from_record(record) for record in result])
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@router.post("/scout", response_model=List[SearchScoutResult])
async def scout_search(request: SearchScoutRequest):
    query_text = request.query.strip()
    if not query_text:
        raise HTTPException(status_code=422, detail="Query cannot be empty")
    limit = max(1, min(request.top_k, 50))
    user_id = request.user_id.strip()
    org_id = request.org_id.strip()
    if not user_id or not org_id:
        raise HTTPException(status_code=422, detail="user_id and org_id are required")
    model_handler = get_global_model_handler()
    try:
        embeddings = await model_handler.embed_text([query_text])
    except Exception as exc:
        logger.error(f"Failed to embed search query: {exc}")
        raise HTTPException(status_code=500, detail="Failed to embed query")
    if not embeddings or not embeddings[0]:
        raise HTTPException(status_code=500, detail="Empty embedding returned for query")
    neo4j_handler = get_neo4j_handler()
    try:
        search_results = await neo4j_handler.search_across_spaces(
            embeddings[0], limit, user_id=user_id, org_id=org_id
        )
    except Exception as exc:
        logger.error(f"Vector search failed: {exc}")
        raise HTTPException(status_code=500, detail="Vector search failed")
    return [SearchScoutResult(**item) for item in search_results]
