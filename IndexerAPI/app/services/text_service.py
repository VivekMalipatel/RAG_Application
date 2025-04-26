import logging
import uuid
from fastapi import BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.file_item import FileItem
from app.schemas.file_schemas import RawTextIngestRequest
from app.processors.text_processor import TextProcessor

logger = logging.getLogger(__name__)

class TextService:
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def process_text(self, request: RawTextIngestRequest, background_tasks: BackgroundTasks):
        text_id = str(uuid.uuid4())
        
        text_item = FileItem(
            id=text_id,
            raw_text=request.text[:1000],
            source=request.source,
            file_type="raw_text",
            metadata=request.metadata,
            status="queued"
        )
        
        self.db.add(text_item)
        await self.db.commit()
        
        background_tasks.add_task(self._process_in_background, text_id, request.text)
        
        return text_item
    
    async def _process_in_background(self, text_id: str, text: str):
        async with AsyncSession() as session:
            text_item = await session.get(FileItem, text_id)
            text_item.status = "processing"
            await session.commit()
        
            try:
                processor = TextProcessor()
                result = await processor.process(text)
                
                text_item.status = "completed"
                text_item.embedding_stored = result.get("embedding_id")
                text_item.metadata = {
                    **(text_item.metadata or {}),
                    "tokens": result.get("token_count"),
                    "summary": result.get("summary")
                }
                await session.commit()
                
            except Exception as e:
                logger.error(f"Error processing text {text_id}: {str(e)}")
                text_item.status = "failed"
                text_item.status_message = str(e)
                await session.commit()