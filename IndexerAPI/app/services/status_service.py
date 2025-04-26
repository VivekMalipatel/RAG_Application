import logging
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.file_item import FileItem
from app.schemas.file_schemas import StatusResponse, StatusEnum

logger = logging.getLogger(__name__)

class StatusService:
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def get_item_status(self, item_id: str) -> StatusResponse:
        stmt = select(FileItem).where(FileItem.id == item_id)
        result = await self.db.execute(stmt)
        item = result.scalar_one_or_none()
        
        if not item:
            return None
        
        result = None
        if item.status == "completed":
            result = {
                "embedding_id": item.embedding_stored,
                "metadata": item.metadata
            }
        
        return StatusResponse(
            id=item.id,
            status=StatusEnum(item.status),
            message=item.status_message,
            result=result,
            created_at=item.created_at,
            updated_at=item.updated_at
        )