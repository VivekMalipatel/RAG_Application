import logging
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.schemas.schemas import StatusResponse
from app.queue import QueueHandler

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/status/{item_id}", response_model=StatusResponse)
async def get_status(
    item_id: str,
    db: AsyncSession = Depends(get_db)
):
    logger.info(f"Status check for item: {item_id}")
    
    queue_handler = QueueHandler(db)
    result = await queue_handler.get_item_status(item_id)
    
    if not result:
        raise HTTPException(status_code=404, detail="Item not found")
    
    return StatusResponse(
        id=result["id"],
        status=result["status"],
        message=result["message"],
        result=result.get("result")
    )