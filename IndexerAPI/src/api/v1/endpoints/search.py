import asyncio
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, HTTPException
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
