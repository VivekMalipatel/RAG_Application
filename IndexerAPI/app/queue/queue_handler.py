import uuid
import logging
import json
from datetime import datetime
from typing import Optional, Dict, Any, List, Union
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from app.models.queue_item import QueueItem
from app.models.file_data import FileData
from app.models.text_data import TextData
from app.models.url_data import URLData
from app.models.failure_queue_item import FailureQueueItem

logger = logging.getLogger(__name__)

class QueueHandler:
    def __init__(self, db_session: AsyncSession):
        self.db_session = db_session
    
    async def enqueue_file(self, file_content: bytes, filename: str, source: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        queue_id = str(uuid.uuid4())
        
        if metadata is None:
            metadata = {}
        if 'filename' not in metadata:
            metadata['filename'] = filename
            
        queue_item = QueueItem(
            id=queue_id,
            source=source,
            item_type="file",
            status="queued",
            indexing_datetime=datetime.now(),
            metadata=json.dumps(metadata)
        )
        
        file_data = FileData(
            queue_id=queue_id,
            content=file_content
        )
        
        self.db_session.add(queue_item)
        self.db_session.add(file_data)
        await self.db_session.commit()
        
        logger.info(f"File enqueued with ID: {queue_id}")
        return queue_id
    
    async def enqueue_url(self, url: str, source: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        queue_id = str(uuid.uuid4())
        queue_item = QueueItem(
            id=queue_id,
            source=source,
            item_type="url",
            status="queued",
            indexing_datetime=datetime.now(),
            metadata=json.dumps(metadata) if metadata else None
        )
        
        url_data = URLData(
            queue_id=queue_id,
            url=url
        )
        
        self.db_session.add(queue_item)
        self.db_session.add(url_data)
        await self.db_session.commit()
        
        logger.info(f"URL enqueued with ID: {queue_id}")
        return queue_id
    
    async def enqueue_text(self, text: str, source: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        queue_id = str(uuid.uuid4())
        queue_item = QueueItem(
            id=queue_id,
            source=source,
            item_type="text",
            status="queued",
            indexing_datetime=datetime.now(),
            metadata=json.dumps(metadata) if metadata else None
        )
        
        text_data = TextData(
            queue_id=queue_id,
            content=text
        )
        
        self.db_session.add(queue_item)
        self.db_session.add(text_data)
        await self.db_session.commit()
        
        logger.info(f"Text enqueued with ID: {queue_id}")
        return queue_id
    
    async def get_next_item(self) -> Optional[QueueItem]:
        stmt = (select(QueueItem)
                .where(QueueItem.status == "queued")
                .order_by(QueueItem.indexing_datetime)
                .limit(1))
        
        result = await self.db_session.execute(stmt)
        queue_item = result.scalar_one_or_none()
        
        if queue_item:
            logger.info(f"Retrieved next queue item: {queue_item.id}")
        
        return queue_item
    
    async def get_item_data(self, queue_item: QueueItem) -> Union[bytes, str]:
        if queue_item.item_type == "file":
            stmt = select(FileData).where(FileData.queue_id == queue_item.id)
            result = await self.db_session.execute(stmt)
            file_data = result.scalar_one_or_none()
            return file_data.content if file_data else None
        elif queue_item.item_type == "url":
            stmt = select(URLData).where(URLData.queue_id == queue_item.id)
            result = await self.db_session.execute(stmt)
            url_data = result.scalar_one_or_none()
            return url_data.url if url_data else None
        elif queue_item.item_type == "text":
            stmt = select(TextData).where(TextData.queue_id == queue_item.id)
            result = await self.db_session.execute(stmt)
            text_data = result.scalar_one_or_none()
            return text_data.content if text_data else None
        return None
    
    async def update_status(self, queue_id: str, status: str, message: Optional[str] = None) -> bool:
        stmt = select(QueueItem).where(QueueItem.id == queue_id)
        result = await self.db_session.execute(stmt)
        queue_item = result.scalar_one_or_none()
        
        if not queue_item:
            logger.error(f"Queue item not found: {queue_id}")
            return False
        
        queue_item.status = status
        if message:
            queue_item.message = message
            
        await self.db_session.commit()
        logger.info(f"Updated status for queue item {queue_id} to {status}")
        return True
    
    async def get_item_status(self, queue_id: str) -> Optional[Dict[str, Any]]:
        stmt = select(QueueItem).where(QueueItem.id == queue_id)
        result = await self.db_session.execute(stmt)
        queue_item = result.scalar_one_or_none()
        
        if not queue_item:
            logger.error(f"Queue item not found: {queue_id}")
            return None
        
        return {
            "id": queue_item.id,
            "status": queue_item.status,
            "message": queue_item.message,
            "item_type": queue_item.item_type,
            "source": queue_item.source,
            "indexing_datetime": queue_item.indexing_datetime.isoformat(),
            "metadata": json.loads(queue_item.metadata) if queue_item.metadata else None
        }
    
    async def list_queue_items(self, status: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        stmt = select(QueueItem)
        if status:
            stmt = stmt.where(QueueItem.status == status)
        stmt = stmt.order_by(QueueItem.indexing_datetime.desc()).limit(limit)
        
        result = await self.db_session.execute(stmt)
        queue_items = result.scalars().all()
        
        return [
            {
                "id": item.id,
                "status": item.status,
                "message": item.message,
                "item_type": item.item_type,
                "source": item.source,
                "indexing_datetime": item.indexing_datetime.isoformat(),
                "metadata": json.loads(item.metadata) if item.metadata else None
            }
            for item in queue_items
        ]
    
    async def move_to_failure_queue(self, queue_id: str, error_message: str) -> bool:
        logger.info(f"Moving queue item {queue_id} to failure queue")
        
        stmt = select(QueueItem).where(QueueItem.id == queue_id)
        result = await self.db_session.execute(stmt)
        queue_item = result.scalar_one_or_none()
        
        if not queue_item:
            logger.error(f"Queue item not found when moving to failure queue: {queue_id}")
            return False

        queue_item.status = "failed"
        queue_item.message = error_message

        stmt = select(FailureQueueItem).where(FailureQueueItem.queue_id == queue_id)
        result = await self.db_session.execute(stmt)
        failure_item = result.scalar_one_or_none()
        
        if failure_item:
            metadata = json.loads(queue_item.metadata) if queue_item.metadata else {}
            retry_count = int(failure_item.retry_count) + 1
            failure_item.retry_count = str(retry_count)
            failure_item.error_message = error_message
            failure_item.failure_datetime = datetime.now()
            
            metadata['last_failure'] = datetime.now().isoformat()
            metadata['error_message'] = error_message
            metadata['retry_count'] = retry_count
            queue_item.metadata = json.dumps(metadata)
        else:
            metadata = json.loads(queue_item.metadata) if queue_item.metadata else {}
            metadata['first_failure'] = datetime.now().isoformat()
            metadata['error_message'] = error_message
            metadata['retry_count'] = 1
            queue_item.metadata = json.dumps(metadata)
            
            failure_item = FailureQueueItem(
                queue_id=queue_id,
                error_message=error_message,
                failure_datetime=datetime.now(),
                retry_count="1"
            )
            self.db_session.add(failure_item)
        
        await self.db_session.commit()
        logger.info(f"Queue item {queue_id} moved to failure queue")
        return True
    
    async def get_failure_queue_items(self, limit: int = 100) -> List[Dict[str, Any]]:
        stmt = (select(QueueItem, FailureQueueItem)
                .join(FailureQueueItem, QueueItem.id == FailureQueueItem.queue_id)
                .order_by(FailureQueueItem.failure_datetime.desc())
                .limit(limit))
                
        result = await self.db_session.execute(stmt)
        items = result.all()
        
        return [
            {
                "id": queue_item.id,
                "status": queue_item.status,
                "message": queue_item.message,
                "item_type": queue_item.item_type,
                "source": queue_item.source,
                "indexing_datetime": queue_item.indexing_datetime.isoformat(),
                "metadata": json.loads(queue_item.metadata) if queue_item.metadata else None,
                "error_message": failure_item.error_message,
                "failure_datetime": failure_item.failure_datetime.isoformat(),
                "retry_count": failure_item.retry_count
            }
            for queue_item, failure_item in items
        ]
        
    async def retry_item_from_failure_queue(self, queue_id: str) -> bool:
        logger.info(f"Retrying queue item {queue_id} from failure queue")
        
        stmt = select(QueueItem).where(QueueItem.id == queue_id)
        result = await self.db_session.execute(stmt)
        queue_item = result.scalar_one_or_none()
        
        if not queue_item:
            logger.error(f"Queue item not found when attempting retry: {queue_id}")
            return False

        queue_item.status = "queued"
        queue_item.message = "Retry after failure"

        metadata = json.loads(queue_item.metadata) if queue_item.metadata else {}
        metadata['last_retry'] = datetime.now().isoformat()
        queue_item.metadata = json.dumps(metadata)
        
        # Don't remove from failure queue table - just keep the record for tracking
        # We'll update the retry count when it fails again if needed
        
        await self.db_session.commit()
        logger.info(f"Queue item {queue_id} returned to processing queue for retry")
        return True