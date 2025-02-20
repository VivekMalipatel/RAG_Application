import asyncio
import logging
from app.core.queue.redis_priority_queue import RedisPriorityQueue
from app.core.db_handler.document_handler import DocumentHandler
from app.core.storage_bin.minio.minio_handler import MinIOHandler

class FileEventProcessor:
    """Processes file events from Redis queues with metadata enrichment"""
    
    def __init__(self):
        self.queue = RedisPriorityQueue()
        self.minio = MinIOHandler()
        self.db = DocumentHandler()

    async def process_events(self):
        """Continuous event processing loop"""
        while True:
            try:
                event = await self.queue.consume_events()
                if event:
                    await self._process_single_event(event)
            except Exception as e:
                logging.error(f"Event processing failed: {str(e)}")
                await asyncio.sleep(1)

    async def _process_single_event(self, event: dict) -> None:
        """Process individual event with metadata enrichment"""
        try:
            # Enrich with DB metadata
            metadata = await self.db.get_document_metadata(event['user_id'], event['path'])
            event.update(metadata)
            
            # Fetch file from MinIO
            file_data = await self.minio.get_object(
                event['bucket'], 
                event['path']
            )
            
            # Prepare for processing
            processing_payload = {
                **event,
                'content': file_data.decode(),
                'processing_type': 'fast' if 'chat' in event['path'] else 'full'
            }
            
            logging.info(f"Processing event {event['event_id']}")
            # TODO: Add processor invocation
            
        except Exception as e:
            logging.error(f"Failed processing event {event.get('event_id', 'unknown')}: {str(e)}")
            await self._handle_failed_event(event, e)

    async def _handle_failed_event(self, event: dict, e) -> None:
        """Handle event processing failures"""
        await self.db.update_event_status(
            event['event_id'],
            'failed',
            str(e)
        )
