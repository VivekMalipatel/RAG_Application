import logging
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.schemas.file_schemas import StatusResponse
from app.services.status_service import StatusService

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/status/{item_id}", response_model=StatusResponse)
async def get_status(
    item_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Get processing status of a submitted item
    """
    logger.info(f"Status check for item: {item_id}")
    
    status_service = StatusService(db)
    result = await status_service.get_item_status(item_id)
    
    if not result:
        raise HTTPException(status_code=404, detail="Item not found")
        
    return result