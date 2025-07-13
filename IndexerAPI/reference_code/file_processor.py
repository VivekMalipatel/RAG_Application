import asyncio
import logging
from core.queue.redis_priority_queue import RedisPriorityQueue
from core.db_handler.document_handler import DocumentHandler
from core.storage_bin.minio.minio_handler import MinIOHandler
from config import settings

from services.file_processor.text_processor import TextProcessor
from services.file_processor.image_processor import ImageProcessor
from services.file_processor.audio_processor import AudioProcessor
from services.file_processor.video_processor import VideoProcessor
from services.file_processor.multimodal_processor import MultimodalProcessor
from services.file_processor.structured_data_processor import StructuredDataProcessor

import magic

class FileEventProcessor:
    """Processes file events from Redis queues with metadata enrichment"""
    
    def __init__(self):
        self.queue = RedisPriorityQueue()
        self.minio = MinIOHandler()
        self.db = DocumentHandler()
        self.mime_detector = magic.Magic(mime=True) 

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
        """Process individual event with metadata enrichment, MIME detection, and routing."""
        try:
            # Enrich event with DB metadata
            metadata = await self.db.get_document_metadata(event['user_id'], event['file_path'])
            if metadata:
                # Convert SQLAlchemy model to dictionary
                metadata_dict = {
                    'document_id': metadata.id,
                    'user_id': metadata.user_id,
                    'file_name': metadata.file_name,
                    'mime_type': metadata.mime_type,
                    'file_size': metadata.file_size,
                    'file_hash': metadata.file_hash,
                    'description': metadata.description,
                    'quadrant_status': metadata.quadrant_status,
                }
                event.update(metadata_dict)
            
            # Fetch file from MinIO
            file_data = await self.minio.fetch_file_from_minio(event.get('file_path'))

            # Detect MIME type after retrieval
            file_data.seek(0)
            sample_bytes = file_data.read(2048)
            file_data.seek(0)  # Reset position for subsequent reads
            detected_mime_type = self.mime_detector.from_buffer(sample_bytes)# Detect from first 2KB
            event["mime_type"] = detected_mime_type

            logging.info(f"Detected MIME type: {detected_mime_type} for file {event.get('file_path')}")

            # Route to appropriate processing pipeline asynchronously
            await self._route_to_processor(event, file_data, detected_mime_type)

        except Exception as e:
            logging.error(f"Failed processing event {event.get('file_event_id', 'unknown')}: {str(e)}")
            await self._handle_failed_event(event, e)

    async def _route_to_processor(self, event: dict, file_data: bytes, mime_type: str):
        """Determine file type based on MIME and call the appropriate processor asynchronously."""
        processor_task = None

        # Text-based processing
        if mime_type in settings.TEXT_MIME_TYPES:
            processor_task = asyncio.create_task(TextProcessor().process(event, file_data))

        # Image-based processing
        elif mime_type in settings.IMAGE_MIME_TYPES:
            processor_task = asyncio.create_task(ImageProcessor().process(event, file_data))

        # Audio-based processing
        elif mime_type in settings.AUDIO_MIME_TYPES:
            processor_task = asyncio.create_task(AudioProcessor().process(event, file_data))

        # Video-based processing
        elif mime_type in settings.VIDEO_MIME_TYPES:
            processor_task = asyncio.create_task(VideoProcessor().process(event, file_data))

        # Multimodal (Text + Images)
        elif mime_type in settings.MULTIMODAL_MIME_TYPES:
            processor_task = asyncio.create_task(TextProcessor().process(event, file_data))

        # Structured Data Processing (CSV, Excel)
        elif mime_type in settings.STRUCTURED_MIME_TYPES:
            processor_task = asyncio.create_task(StructuredDataProcessor().process(event, file_data))

        # If no processor is found, log and skip processing
        if processor_task is None:
            logging.warning(f"Unsupported MIME type {mime_type} for {event['file_path']}. Skipping processing.")
            return

        # Wait for processing completion asynchronously
        await processor_task

    async def _handle_failed_event(self, event: dict, e) -> None:
        """Handle event processing failures."""
        await self.db.update_event_status(event['file_event_id'], 'failed', str(e))