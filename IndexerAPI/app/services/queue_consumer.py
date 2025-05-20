import logging
import json
import hashlib
import asyncio
from typing import Dict, Any, List, Optional

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
                filename = metadata.get('filename', f"unknown_{queue_item.id}")
                stable_doc_id_str = f"{queue_item.source}:{filename}"
                stable_doc_id = hashlib.sha256(stable_doc_id_str.encode()).hexdigest()
                
                processor_generator = processor.process(data, metadata, source=queue_item.source)
                
                processed_items = []
                all_embeddings = []
                all_metadata = []
                embedding_tasks = []
                item_batches = []
                task_results = {}
                next_task_id = 0
                
                image_embedding_batch_size = 3
                text_embedding_batch_size = 35
                
                logger.info(f"Using batch sizes: Image={image_embedding_batch_size}, Text={text_embedding_batch_size}")
                
                current_image_batch = []
                current_text_batch = []
                
                async for item in processor_generator:
                    try:
                        item_metadata = item.get("metadata", {}).copy()
                        item_metadata.update({
                            "source": queue_item.source,
                            "queue_id": queue_item.id,
                            "timestamp": str(queue_item.indexing_datetime),
                            "document_id": stable_doc_id,
                        })
                        
                        if "page" in item:
                            item_metadata["page_number"] = item["page"]
                            item_metadata["content_type"] = "image"
                        elif "batch" in item:
                            item_metadata["batch_number"] = item["batch"]
                            item_metadata["content_type"] = "text"
                        else:
                            item_metadata["content_type"] = "text"
                        
                        if "image_b64" in item:
                            current_image_batch.append(item)
                            
                            if len(current_image_batch) >= image_embedding_batch_size:
                                image_items = current_image_batch.copy()
                                task_id = next_task_id
                                next_task_id += 1
                                task = asyncio.create_task(self._process_embedding_batch(image_items, False, queue_item.id))
                                task_results[task_id] = task
                                item_batches.append((task_id, "image", image_items))
                                current_image_batch = []
                        else:
                            current_text_batch.append(item)
                            
                            if len(current_text_batch) >= text_embedding_batch_size:
                                text_items = current_text_batch.copy()
                                task_id = next_task_id
                                next_task_id += 1
                                task = asyncio.create_task(self._process_embedding_batch(text_items, True, queue_item.id))
                                task_results[task_id] = task
                                item_batches.append((task_id, "text", text_items))
                                current_text_batch = []
                    except Exception as item_e:
                        logger.error(f"Error processing item {len(processed_items) + 1} for queue item {queue_item.id}: {str(item_e)}")
                
                if current_image_batch:
                    image_items = current_image_batch.copy()
                    task_id = next_task_id
                    next_task_id += 1
                    task = asyncio.create_task(self._process_embedding_batch(image_items, False, queue_item.id))
                    task_results[task_id] = task
                    item_batches.append((task_id, "image", image_items))
                
                if current_text_batch:
                    text_items = current_text_batch.copy()
                    task_id = next_task_id
                    next_task_id += 1
                    task = asyncio.create_task(self._process_embedding_batch(text_items, True, queue_item.id))
                    task_results[task_id] = task
                    item_batches.append((task_id, "text", text_items))

                for task_id, batch_type, batch_items in item_batches:
                    logger.info(f"Awaiting embeddings for batch {task_id} ({batch_type}, {len(batch_items)} items)")
                    batch_embeddings = await task_results[task_id]
                    logger.info(f"Received embeddings for batch {task_id} - {len(batch_embeddings) if batch_embeddings else 0} vectors")
                    
                    for batch_item, embedding in zip(batch_items, batch_embeddings):
                        item_meta = batch_item.get("metadata", {}).copy()
                        item_meta.update({
                            "source": queue_item.source,
                            "queue_id": queue_item.id,
                            "timestamp": str(queue_item.indexing_datetime),
                            "document_id": stable_doc_id,
                            "content_type": batch_type
                        })
                        
                        if batch_type == "image":
                            item_meta["page_number"] = batch_item.get("page")
                        else:
                            item_meta["batch_number"] = batch_item.get("batch")
                        
                        all_embeddings.append(embedding)
                        all_metadata.append(item_meta)
                        
                        processed_items.append({
                            "text": batch_item.get("text", ""),
                            "metadata": item_meta,
                            "embeddings": embedding
                        })
                
                #TODO: Save page by page instead of all at once
                if all_embeddings:
                    try:
                        document_metadata = {
                            "source": queue_item.source,
                            "queue_id": queue_item.id,
                            "timestamp": str(queue_item.indexing_datetime),
                            "filename": metadata.get("filename", ""),
                            "file_type": metadata.get("file_type", ""),
                            "total_items": len(processed_items)
                        }
                        
                        if all_metadata:
                            document_metadata["content_type"] = all_metadata[0]["content_type"]
                        
                        for key, value in metadata.items():
                            if key not in document_metadata:
                                document_metadata[key] = value
                        
                        document_metadata["pages"] = all_metadata          

                        removed = self.vector_store.remove_document(stable_doc_id)
                        if removed:
                            logger.info(f"Removed existing document version for {stable_doc_id}")
                        
                        vectors_added = self.vector_store.add_document(
                            doc_id=stable_doc_id,
                            embeddings=all_embeddings,
                            metadata=document_metadata
                        )
                        
                        logger.info(f"Added {vectors_added} vectors to vector store for document {stable_doc_id}")

                        if self.vector_store.index and self.vector_store.index.ntotal % 100 == 0:
                            self.vector_store.save()
                            
                    except Exception as e:
                        logger.error(f"Error adding document to vector store: {str(e)}")
                
                await self.queue_handler.update_status(queue_item.id, "completed", f"Processing completed: {len(processed_items)} items indexed")
                
                return {
                    "id": queue_item.id,
                    "source": queue_item.source,
                    "status": "completed",
                    "metadata": metadata,
                    "document_id": stable_doc_id,
                    "total_items": len(processed_items),
                    "total_vectors": len(all_embeddings)
                }
                
            except Exception as e:
                error_msg = f"Error processing item {queue_item.id}: {str(e)}"
                logger.error(error_msg)
                await self.queue_handler.move_to_failure_queue(queue_item.id, error_msg)
                return {"id": queue_item.id, "status": "failed", "error": error_msg}
            
        except Exception as e:
            error_msg = f"Unexpected error processing item {queue_item.id}: {str(e)}"
            logger.error(error_msg)
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
                    
                    if result.get("status") != "failed":
                        logger.info(f"Successfully processed item {result['id']} with {result.get('total_vectors', 0)} vectors")
                else:
                    logger.info(f"No items in queue, waiting {poll_interval} seconds")
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
            logger.error(f"Error saving vector store: {str(e)}")
    
    def stop_processing(self):
        logger.info("Stopping queue processing loop")
        self.running = False
        
    async def list_failure_queue(self, limit: int = 100) -> List[Dict[str, Any]]:
        return await self.queue_handler.get_failure_queue_items(limit)
    
    async def retry_failed_item(self, queue_id: str) -> bool:
        return await self.queue_handler.retry_item_from_failure_queue(queue_id)
    
    async def _process_embedding_batch(self, items, is_text_batch, queue_id):
        try:
            if is_text_batch:
                texts = [item["text"] for item in items]
                embeddings = await self.model_handler.embed_text(texts)
            else:
                images = []
                for item in items:
                    if "image_b64" in item:
                        images.append({"image": item["image_b64"], "text": item.get("text", "")})
                
                if images:
                    embeddings = await self.model_handler.embed_image(images)

                else:
                    embeddings = []
            
            return embeddings
        except Exception as e:
            logger.error(f"Error processing batch embeddings: {str(e)}", exc_info=True)
            return [None] * len(items)