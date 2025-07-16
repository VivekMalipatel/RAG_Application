from fastapi import APIRouter, HTTPException
from typing import Optional, Dict, Any, List
from pydantic import BaseModel

from core.storage.neo4j_handler import get_neo4j_handler

router = APIRouter()

class CypherQueryRequest(BaseModel):
    query: str
    parameters: Optional[Dict[str, Any]] = None

@router.post("/cypher", response_model=List[Dict[str, Any]])
async def execute_cypher_query(request: CypherQueryRequest):
    try:
        neo4j_handler = get_neo4j_handler()
        result = await neo4j_handler.execute_cypher_query(
            query=request.query,
            parameters=request.parameters
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))