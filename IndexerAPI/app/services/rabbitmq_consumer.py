import asyncio
import logging
import hashlib
import json
from typing import Optional

from config import settings

from core.queue.rabbitmq_handler import rabbitmq_handler
from core.queue.task_types import TaskMessage, TaskType
from core.model.model_handler import ModelHandler, get_global_model_handler
from services.vector_store import VectorStore, get_global_vector_store

logger = logging.getLogger(__name__)

_global_rabbitmq_consumer: Optional['RabbitMQConsumer'] = None

class RabbitMQConsumer:
    def __init__(self, model_handler: ModelHandler = None, vector_store: VectorStore = None):
        self.model_handler = model_handler or get_global_model_handler()
        self.vector_store = vector_store or get_global_vector_store()
        self.processors = {}
        self.is_running = False
        self._consumer_task = None
        self.vector_store.load()
        logger.info("RabbitMQConsumer initialized with FAISS vector store")

    def register_processor(self, processor_type: str, processor):
        self.processors[processor_type] = processor
        logger.info(f"Registered {processor.__class__.__name__} for item type: {processor_type}")

    async def start_processing(self):
        if self.is_running:
            logger.warning("Consumer is already running")
            return

        await rabbitmq_handler.connect()
        self.is_running = True
        logger.info(f"Starting RabbitMQ consumer with max concurrency: {settings.MAX_CONCURRENCY}")
        self._consumer_task = asyncio.create_task(self._consume_loop())

    async def _consume_loop(self):
        tasks = set()
        logger.info("Consumer loop started")
        while self.is_running:
            try:
                incoming_message = await self._get_next_message()
                if incoming_message is None:
                    await asyncio.sleep(0.2)
                    continue

                task = asyncio.create_task(self._handle_message(incoming_message))
                tasks.add(task)
                tasks = {t for t in tasks if not t.done()}
            except Exception as e:
                logger.error(f"Error in consumer loop: {str(e)}")
                await asyncio.sleep(1)

    async def _get_next_message(self):
        try:
            await rabbitmq_handler._ensure_connected()
            message = await rabbitmq_handler.queue.get(no_ack=False, fail=False)
            if message is None:
                return None
            return message
        except Exception as e:
            logger.error(f"Error getting message from queue: {str(e)}")
            return None

    async def _handle_message(self, message):
        try:
            try:
                message_data = json.loads(message.body.decode())
                task_message = TaskMessage.from_dict(message_data)
                logger.info(f"Processing task: {task_message.task_id} of type {task_message.task_type.value}")
                
                await self._process_message(task_message)
                
                message.ack()
                logger.info(f"Task {task_message.task_id} processed successfully and acknowledged")
                
            except Exception as e:
                logger.error(f"Error processing message: {str(e)}")
                message.reject(requeue=False)
                logger.error(f"Task {task_message.task_id if 'task_message' in locals() else 'unknown'} rejected and sent to dead letter queue")
                raise
                
        except Exception:
            pass

    async def stop_processing(self):
        if not self.is_running:
            return
        self.is_running = False
        if self._consumer_task:
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except asyncio.CancelledError:
                logger.info("Consumer task cancelled")
        await rabbitmq_handler.disconnect()
        logger.info("RabbitMQ consumer stopped")

    async def _process_message(self, task_message: TaskMessage):
        try:
            if task_message.task_type == TaskType.FILE:
                await self._process_file_task(task_message)
            elif task_message.task_type == TaskType.URL:
                await self._process_url_task(task_message)
            elif task_message.task_type == TaskType.TEXT:
                await self._process_text_task(task_message)
            else:
                logger.error(f"Unknown task type: {task_message.task_type}")
                raise ValueError(f"Unknown task type: {task_message.task_type}")
                
        except Exception as e:
            logger.error(f"Error processing task {task_message.task_id}: {str(e)}")
            raise

    async def _process_file_task(self, task_message: TaskMessage):
        if "file" not in self.processors:
            raise ValueError("File processor not registered")
        
        processor = self.processors["file"]
        
        try:
            logger.info(f"Processing file task {task_message.task_id}: {task_message.filename}")
            
            metadata = task_message.metadata or {}
            filename = metadata.get('filename', task_message.filename or f"unknown_{task_message.task_id}")
            stable_doc_id_str = f"{task_message.source}:{filename}"
            stable_doc_id = hashlib.sha256(stable_doc_id_str.encode()).hexdigest()
            
            processor_generator = processor.process(
                data=task_message.file_content,
                metadata=metadata,
                source=task_message.source
            )
            
            await self._process_items(processor_generator, task_message, stable_doc_id)
            
            logger.info(f"File task {task_message.task_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Error processing file task {task_message.task_id}: {str(e)}")
            raise

    async def _process_url_task(self, task_message: TaskMessage):
        if "url" not in self.processors:
            raise ValueError("URL processor not registered")
        
        processor = self.processors["url"]
        
        try:
            logger.info(f"Processing URL task {task_message.task_id}: {task_message.url}")
            
            metadata = task_message.metadata or {}
            stable_doc_id_str = f"{task_message.source}:{task_message.url}"
            stable_doc_id = hashlib.sha256(stable_doc_id_str.encode()).hexdigest()
            
            processor_generator = processor.process(
                data=task_message.url,
                metadata=metadata,
                source=task_message.source
            )
            
            await self._process_items(processor_generator, task_message, stable_doc_id)
            
            logger.info(f"URL task {task_message.task_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Error processing URL task {task_message.task_id}: {str(e)}")
            raise

    async def _process_text_task(self, task_message: TaskMessage):
        if "text" not in self.processors:
            raise ValueError("Text processor not registered")
        
        processor = self.processors["text"]
        
        try:
            logger.info(f"Processing text task {task_message.task_id}")
            
            metadata = task_message.metadata or {}
            stable_doc_id_str = f"{task_message.source}:text_{task_message.task_id}"
            stable_doc_id = hashlib.sha256(stable_doc_id_str.encode()).hexdigest()
            
            processor_generator = processor.process(
                data=task_message.text_content,
                metadata=metadata,
                source=task_message.source
            )
            
            await self._process_items(processor_generator, task_message, stable_doc_id)
            
            logger.info(f"Text task {task_message.task_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Error processing text task {task_message.task_id}: {str(e)}")
            raise

    async def _process_items(self, processor_generator, task_message: TaskMessage, stable_doc_id: str):
        try:
            processed_items = []
            all_embeddings = []
            all_metadata = []
            
            items = []
            async for item in processor_generator:
                items.append(item)
            
            if not items:
                logger.info(f"No items to process for task {task_message.task_id}")
                return
            
            logger.info(f"Processing {len(items)} items concurrently for task {task_message.task_id}")
            
            embedding_tasks = []
            for item in items:
                if "image_b64" in item:
                    image_data = [{"image": item["image_b64"], "text": item.get("text", "")}]
                    task = self.model_handler.embed_image(image_data)
                else:
                    task = self.model_handler.embed_text([item.get("text", "")])
                
                embedding_tasks.append(task)
            
            logger.info(f"Running {len(embedding_tasks)} embedding tasks concurrently")
            embeddings_results = await asyncio.gather(*embedding_tasks, return_exceptions=True)

            for i, (item, embedding_result) in enumerate(zip(items, embeddings_results)):
                try:
                    if isinstance(embedding_result, Exception):
                        logger.error(f"Embedding failed for item {i}: {str(embedding_result)}")
                        continue
                    
                    embedding = embedding_result[0] if embedding_result else None
                    if embedding is None:
                        logger.warning(f"No embedding generated for item {i}")
                        continue

                    item_metadata = item.get("metadata", {}).copy()
                    item_metadata.update({
                        "source": task_message.source,
                        "task_id": task_message.task_id,
                        "timestamp": str(task_message.created_at),
                        "document_id": stable_doc_id,
                    })
                    
                    if task_message.filename:
                        item_metadata["filename"] = task_message.filename
                    if task_message.url:
                        item_metadata["url"] = task_message.url
                    
                    if "page" in item:
                        item_metadata["page_number"] = item["page"]
                        item_metadata["content_type"] = "image"
                    elif "batch" in item:
                        item_metadata["batch_number"] = item["batch"]
                        item_metadata["content_type"] = "text"
                    else:
                        item_metadata["content_type"] = "text"
                    
                    all_embeddings.append(embedding)
                    all_metadata.append(item_metadata)
                    
                    processed_items.append({
                        "text": item.get("text", ""),
                        "metadata": item_metadata,
                        "embeddings": embedding
                    })
                    
                except Exception as item_e:
                    logger.error(f"Error processing item {i} for task {task_message.task_id}: {str(item_e)}")
            
            if all_embeddings:
                try:
                    document_metadata = {
                        "source": task_message.source,
                        "task_id": task_message.task_id,
                        "timestamp": str(task_message.created_at),
                        "total_items": len(processed_items)
                    }
                    
                    if task_message.filename:
                        document_metadata["filename"] = task_message.filename
                    if task_message.url:
                        document_metadata["url"] = task_message.url
                    
                    metadata = task_message.metadata or {}
                    for key, value in metadata.items():
                        if key not in document_metadata:
                            document_metadata[key] = value
                    
                    if all_metadata:
                        document_metadata["content_type"] = all_metadata[0]["content_type"]
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
                    raise
            
            logger.info(f"Stored {len(processed_items)} items for task {task_message.task_id}")
            
        except Exception as e:
            logger.error(f"Error in processing for task {task_message.task_id}: {str(e)}")
            raise

    async def get_queue_info(self):
        return await rabbitmq_handler.get_queue_info()

    async def purge_queue(self):
        await rabbitmq_handler.purge_queue()

def get_global_rabbitmq_consumer() -> RabbitMQConsumer:
    global _global_rabbitmq_consumer
    if _global_rabbitmq_consumer is None:
        _global_rabbitmq_consumer = RabbitMQConsumer()
    return _global_rabbitmq_consumer

async def cleanup_global_rabbitmq_consumer():
    global _global_rabbitmq_consumer
    if _global_rabbitmq_consumer:
        await _global_rabbitmq_consumer.stop_processing()
        try:
            if (hasattr(_global_rabbitmq_consumer.vector_store, 'index') and 
                _global_rabbitmq_consumer.vector_store.index is not None and 
                _global_rabbitmq_consumer.vector_store.index.ntotal > 0):
                logger.info("Saving vector store during shutdown")
                _global_rabbitmq_consumer.vector_store.save()
            else:
                logger.info("No vector data to save during shutdown")
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
        _global_rabbitmq_consumer = None
