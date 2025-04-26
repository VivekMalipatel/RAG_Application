import logging
import uuid
from fastapi import BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.file_item import FileItem
from app.schemas.file_schemas import UrlIngestRequest
from app.processors.url_processor import UrlProcessor

logger = logging.getLogger(__name__)

class UrlService:
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def process_url(self, request: UrlIngestRequest, background_tasks: BackgroundTasks):
        url_id = str(uuid.uuid4())
        
        url_item = FileItem(
            id=url_id,
            url=str(request.url),
            source=request.source,
            file_type="url",
            metadata=request.metadata,
            status="queued"
        )
        
        self.db.add(url_item)
        await self.db.commit()
        
        background_tasks.add_task(self._process_in_background, url_id, str(request.url))
        
        return url_item
    
    async def _process_in_background(self, url_id: str, url: str):
        async with AsyncSession() as session:
            url_item = await session.get(FileItem, url_id)
            url_item.status = "processing"
            await session.commit()
        
            try:
                is_youtube = "youtube.com" in url or "youtu.be" in url
                
                processor = UrlProcessor(is_youtube=is_youtube)
                
                result = await processor.process(url)
                
                url_item.status = "completed"
                url_item.embedding_stored = result.get("embedding_id")
                url_item.metadata = {
                    **(url_item.metadata or {}),
                    "extracted_content": result.get("content_summary")
                }
                await session.commit()
                
            except Exception as e:
                logger.error(f"Error processing URL {url_id}: {str(e)}")
                url_item.status = "failed"
                url_item.status_message = str(e)
                await session.commit()