import logging
import json
import hashlib
from typing import Dict, Any, List, Optional, Union
import asyncio
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from app.queue.queue_handler import QueueHandler
from app.core.model.model_handler import ModelHandler
from app.models.queue_item import QueueItem
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)

class QueueConsumer:
    def __init__(self, db_session: AsyncSession, model_handler: ModelHandler = None, vector_store: VectorStore = None):
        self.db_session = db_session
        self.queue_handler = QueueHandler(db_session)
        self.model_handler = model_handler or ModelHandler()
        self.running = False
        self.processors = {}
        self.vector_store = vector_store
        self.vector_store.load()
        logger.info("QueueConsumer initialized with FAISS vector store")
    
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
                await self.queue_handler.move_to_failure_queue(queue_item.id, error_msg)
                return {"id": queue_item.id, "status": "failed", "error": error_msg}
            
            data = await self.queue_handler.get_item_data(queue_item)
            if data is None:
                error_msg = f"No data found for queue item: {queue_item.id}"
                logger.error(error_msg)
                await self.queue_handler.move_to_failure_queue(queue_item.id, error_msg)
                return {"id": queue_item.id, "status": "failed", "error": error_msg}
            
            metadata = json.loads(queue_item.item_metadata) if queue_item.item_metadata else {}
            try:
                processed_data = await processor.process(data, metadata, source=queue_item.source)
            except Exception as e:
                error_msg = f"Error processing item {queue_item.id}: {str(e)}"
                logger.error(error_msg, exc_info=True)
                await self.queue_handler.move_to_failure_queue(queue_item.id, error_msg)
                return {"id": queue_item.id, "status": "failed", "error": error_msg}
            
            embeddings = []
            if "data" in processed_data:
                if isinstance(processed_data["data"], list):
                    first_item = processed_data["data"][0] if processed_data["data"] else None
                    
                    if first_item and isinstance(first_item, dict) and "image" in first_item:
                        logger.info(f"Generating image embeddings for {queue_item.id}")
                        try:
                            embeddings = await self.model_handler.embed_image(processed_data["data"])
                        except Exception as e:
                            logger.error(f"Error generating image embeddings: {str(e)}", exc_info=True)
                            error_msg = f"Error generating embeddings: {str(e)}"
                            await self.queue_handler.move_to_failure_queue(queue_item.id, error_msg)
                            return {"id": queue_item.id, "status": "failed", "error": error_msg}
                    elif first_item and isinstance(first_item, str):
                        logger.info(f"Generating text embeddings for {queue_item.id}")
                        try:
                            embeddings = await self.model_handler.embed_text(processed_data["data"])
                        except Exception as e:
                            logger.error(f"Error generating text embeddings: {str(e)}", exc_info=True)
                            error_msg = f"Error generating embeddings: {str(e)}"
                            await self.queue_handler.move_to_failure_queue(queue_item.id, error_msg)
                            return {"id": queue_item.id, "status": "failed", "error": error_msg}

            result = {
                "id": queue_item.id,
                "source": queue_item.source,
                "metadata": processed_data.get("metadata", metadata),
                "timestamp": queue_item.indexing_datetime,
                "data": processed_data.get("data", []),
                "embeddings": embeddings
            }
            
            await self.queue_handler.update_status(queue_item.id, "completed", "Processing completed")
            logger.info(f"Successfully processed queue item {queue_item.id}")
            
            return result
            
        except Exception as e:
            error_msg = f"Unexpected error processing item {queue_item.id}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            await self.queue_handler.move_to_failure_queue(queue_item.id, error_msg)
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
                    try:
                        await asyncio.sleep(poll_interval)
                    except asyncio.CancelledError:
                        logger.info("Processing task cancelled during sleep")
                        self._save_vector_store_if_needed()
                        raise
                    
            except asyncio.CancelledError:
                logger.info("Processing task cancelled")
                self.running = False
                self._save_vector_store_if_needed()
                raise
            except Exception as e:
                logger.error(f"Error in processing loop: {str(e)}", exc_info=True)
                try:
                    await asyncio.sleep(poll_interval)
                except asyncio.CancelledError:
                    logger.info("Processing task cancelled during error recovery")
                    self._save_vector_store_if_needed()
                    raise
    
    def _save_vector_store_if_needed(self):
        try:
            if hasattr(self.vector_store, 'index') and self.vector_store.index is not None and self.vector_store.index.ntotal > 0:
                logger.info("Saving vector store during shutdown")
                self.vector_store.save()
            else:
                logger.info("No vector data to save during shutdown")
        except Exception as e:
            logger.error(f"Error saving vector store during shutdown: {str(e)}", exc_info=True)
    
    def stop_processing(self):
        logger.info("Stopping queue processing loop")
        self.running = False
    
    def _handle_processed_data(self, result: Dict[str, Any]):
        if result.get("status") == "failed":
            logger.error(f"Failed to process item {result['id']}: {result.get('error')}")
            return
        
        embeddings = result.get("embeddings", [])
        if not embeddings:
            logger.warning(f"No embeddings found for item {result['id']}")
            return
            
        data_items = result.get("data", [])
        
        logger.info(f"Processing item {result['id']} with embeddings for {len(embeddings)} items")
        
        try:
            
            metadata = {
                "source": result.get("source"),
                "queue_id": result.get("id"),
                "timestamp": str(result.get("timestamp")),
                "page_count": len(embeddings)
            }
            
            first_item = data_items[0] if data_items else None
            if first_item and isinstance(first_item, dict) and "image" in first_item:
                metadata["content_type"] = "image"
            else:
                metadata["content_type"] = "text"
                
            if isinstance(result.get("metadata"), dict):
                result_metadata = result.get("metadata")
                reserved_keys = metadata.keys()
                for key, value in result_metadata.items():
                    if key in reserved_keys:
                        new_key = f"data_{key}"
                        metadata[new_key] = value
                    else:
                        metadata[key] = value
            
            filename = metadata.get('filename', f"item_{result['id']}")
            doc_id = result['id']
            
            removed = self.vector_store.remove_document(doc_id)
            if removed:
                logger.info(f"Removed existing document version for {doc_id} ({result.get('source')}/{filename})")
            else:
                logger.info(f"No existing document found for {doc_id} ({result.get('source')}/{filename})")
            
            vectors_added = self.vector_store.add_document(
                doc_id=doc_id,
                embeddings=embeddings,
                metadata=metadata
            )
            
            logger.info(f"Added {vectors_added} vectors to FAISS index for document {stable_doc_id} across {len(embeddings)} pages")
            
            if self.vector_store.index and self.vector_store.index.ntotal % 100 == 0:
                logger.info("Saving FAISS index to disk...")
                self.vector_store.save()
            
        except Exception as e:
            logger.error(f"Error storing embeddings in vector database: {str(e)}", exc_info=True)
        
    async def list_failure_queue(self, limit: int = 100) -> List[Dict[str, Any]]:
        return await self.queue_handler.get_failure_queue_items(limit)
    
    async def retry_failed_item(self, queue_id: str) -> bool:
        return await self.queue_handler.retry_item_from_failure_queue(queue_id)