import asyncio
import logging
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession
from app.queue.rabbitmq_handler import rabbitmq_handler, TaskMessage, TaskType
from app.core.model.model_handler import ModelHandler
from app.services.vector_store import VectorStore
from app.processors.file_processor import FileProcessor
from app.processors.url_processor import URLProcessor
from app.processors.raw_text_processor import RawTextProcessor

logger = logging.getLogger(__name__)


class RabbitMQConsumer:
    def __init__(self, db_session: AsyncSession, model_handler: ModelHandler, vector_store: VectorStore):
        self.db_session = db_session
        self.model_handler = model_handler
        self.vector_store = vector_store
        self.processors = {}
        self.is_running = False
        self._consumer_task = None

    def register_processor(self, processor_type: str, processor):
        """Register a processor for a specific task type"""
        self.processors[processor_type] = processor
        logger.info(f"Registered processor for type: {processor_type}")

    async def start_processing(self):
        """Start consuming messages from RabbitMQ"""
        if self.is_running:
            logger.warning("Consumer is already running")
            return

        try:
            await rabbitmq_handler.connect()
            self.is_running = True
            
            logger.info("Starting RabbitMQ message consumption...")
            await rabbitmq_handler.consume_messages(self._process_message)
            
        except Exception as e:
            logger.error(f"Error starting RabbitMQ consumer: {str(e)}")
            self.is_running = False
            raise

    async def stop_processing(self):
        """Stop consuming messages"""
        if not self.is_running:
            return

        self.is_running = False
        await rabbitmq_handler.disconnect()
        logger.info("RabbitMQ consumer stopped")

    async def _process_message(self, task_message: TaskMessage):
        """Process a single task message"""
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
            # Let the exception propagate so RabbitMQ can handle retries
            raise

    async def _process_file_task(self, task_message: TaskMessage):
        """Process a file task"""
        if "file" not in self.processors:
            raise ValueError("File processor not registered")
        
        processor = self.processors["file"]
        
        try:
            logger.info(f"Processing file task {task_message.task_id}: {task_message.filename}")
            
            # Process the file
            async for _ in processor.process(
                task_message.file_content,
                metadata=task_message.metadata,
                source=task_message.source
            ):
                pass
            
            logger.info(f"File task {task_message.task_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Error processing file task {task_message.task_id}: {str(e)}")
            raise

    async def _process_url_task(self, task_message: TaskMessage):
        """Process a URL task"""
        if "url" not in self.processors:
            raise ValueError("URL processor not registered")
        
        processor = self.processors["url"]
        
        try:
            logger.info(f"Processing URL task {task_message.task_id}: {task_message.url}")
            
            # Process the URL
            async for _ in processor.process(
                task_message.url,
                source=task_message.source,
                metadata=task_message.metadata
            ):
                pass
            logger.info(f"URL task {task_message.task_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Error processing URL task {task_message.task_id}: {str(e)}")
            raise

    async def _process_text_task(self, task_message: TaskMessage):
        """Process a text task"""
        if "text" not in self.processors:
            raise ValueError("Text processor not registered")
        
        processor = self.processors["text"]
        
        try:
            logger.info(f"Processing text task {task_message.task_id}")
            
            # Process the text
            async for _ in processor.process(
                task_message.text_content,
                source=task_message.source,
                metadata=task_message.metadata
            ):
                pass
            
            logger.info(f"Text task {task_message.task_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Error processing text task {task_message.task_id}: {str(e)}")
            raise

    async def get_queue_info(self):
        """Get current queue information"""
        return await rabbitmq_handler.get_queue_info()

    async def purge_queue(self):
        """Purge all messages from the queue"""
        await rabbitmq_handler.purge_queue()
