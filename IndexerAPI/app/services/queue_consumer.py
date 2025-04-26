import logging
import json
from typing import Dict, Any, List, Optional, Union
import asyncio
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from app.queue.queue_handler import QueueHandler
from app.core.model.model_handler import ModelHandler
from app.models.queue_item import QueueItem

logger = logging.getLogger(__name__)

class QueueConsumer:
    def __init__(self, db_session: AsyncSession, model_handler: ModelHandler = None):
        self.db_session = db_session
        self.queue_handler = QueueHandler(db_session)
        self.model_handler = model_handler or ModelHandler()
        self.running = False
        self.processors = {}
        logger.info("QueueConsumer initialized")
    
    def register_processor(self, item_type: str, processor):
        self.processors[item_type] = processor
        logger.info(f"Registered {processor.__class__.__name__} for item type: {item_type}")
    
    async def process_queue_item(self, queue_item: QueueItem) -> Dict[str, Any]:
        try:
            logger.info(f"Processing queue item {queue_item.id} of type {queue_item.item_type}")
            
            await self.queue_handler.update_status(queue_item.id, "processing")
            
            processor = self.processors.get(queue_item.item_type)
            if not processor:
                error_msg = f"No processor registered for item type: {queue_item.item_type}"
                logger.error(error_msg)
                await self.queue_handler.update_status(queue_item.id, "failed", error_msg)
                return {"id": queue_item.id, "status": "failed", "error": error_msg}
            
            data = await self.queue_handler.get_item_data(queue_item)
            if data is None:
                error_msg = f"No data found for queue item: {queue_item.id}"
                logger.error(error_msg)
                await self.queue_handler.update_status(queue_item.id, "failed", error_msg)
                return {"id": queue_item.id, "status": "failed", "error": error_msg}
            
            metadata = json.loads(queue_item.metadata) if queue_item.metadata else {}
            try:
                processed_data = await processor.process(data, metadata)
            except Exception as e:
                error_msg = f"Error processing item {queue_item.id}: {str(e)}"
                logger.error(error_msg, exc_info=True)
                await self.queue_handler.update_status(queue_item.id, "failed", error_msg)
                return {"id": queue_item.id, "status": "failed", "error": error_msg}
            
            embeddings = []
            if "data" in processed_data:
                if isinstance(processed_data["data"], list):
                    first_item = processed_data["data"][0] if processed_data["data"] else None
                    
                    if first_item and isinstance(first_item, dict) and "image" in first_item:
                        logger.info(f"Generating image embeddings for {queue_item.id}")
                        try:
                            embeddings = self.model_handler.embed_image(processed_data["data"])
                        except Exception as e:
                            logger.error(f"Error generating image embeddings: {str(e)}", exc_info=True)
                            error_msg = f"Error generating embeddings: {str(e)}"
                            await self.queue_handler.update_status(queue_item.id, "failed", error_msg)
                            return {"id": queue_item.id, "status": "failed", "error": error_msg}
                    elif first_item and isinstance(first_item, str):
                        logger.info(f"Generating text embeddings for {queue_item.id}")
                        try:
                            embeddings = self.model_handler.embed_text(processed_data["data"])
                        except Exception as e:
                            logger.error(f"Error generating text embeddings: {str(e)}", exc_info=True)
                            error_msg = f"Error generating embeddings: {str(e)}"
                            await self.queue_handler.update_status(queue_item.id, "failed", error_msg)
                            return {"id": queue_item.id, "status": "failed", "error": error_msg}

            result = {
                "id": queue_item.id,
                "source": processed_data.get("source", queue_item.source),
                "metadata": processed_data.get("metadata", metadata),
                "timestamp": processed_data.get("timestamp", datetime.now().isoformat()),
                "data": processed_data.get("data", []),
                "embeddings": embeddings
            }
            
            await self.queue_handler.update_status(queue_item.id, "completed", "Processing completed")
            logger.info(f"Successfully processed queue item {queue_item.id}")
            
            return result
            
        except Exception as e:
            error_msg = f"Unexpected error processing item {queue_item.id}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            await self.queue_handler.update_status(queue_item.id, "failed", error_msg)
            return {"id": queue_item.id, "status": "failed", "error": error_msg}
    
    async def start_processing(self, poll_interval: float = 2.0):
        self.running = True
        logger.info("Starting queue processing loop")
        
        while self.running:
            try:
                queue_item = await self.queue_handler.get_next_item()
                
                if queue_item:
                    logger.info(f"Found queue item {queue_item.id} to process")
                    result = await self.process_queue_item(queue_item)
                    
                    logger.info(f"Processed item {result['id']} with {len(result.get('embeddings', []))} embeddings")
                    
                    self._handle_processed_data(result)
                else:
                    logger.debug(f"No items in queue, waiting {poll_interval} seconds")
                    await asyncio.sleep(poll_interval)
                    
            except Exception as e:
                logger.error(f"Error in processing loop: {str(e)}", exc_info=True)
                await asyncio.sleep(poll_interval)
    
    def stop_processing(self):
        logger.info("Stopping queue processing loop")
        self.running = False
    
    def _handle_processed_data(self, result: Dict[str, Any]):
        if result.get("status") == "failed":
            logger.error(f"Failed to process item {result['id']}: {result.get('error')}")
            return
        
        embedding_count = len(result.get("embeddings", []))
        data_count = len(result.get("data", []))
        
        logger.info(f"Successfully processed item {result['id']} with {embedding_count} embeddings for {data_count} data items")

        #TODO: Implement any additional logic for handling processed data (Sending it to vector DB, etc.)