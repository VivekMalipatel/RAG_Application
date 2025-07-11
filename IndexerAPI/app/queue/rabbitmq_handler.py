import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum

import aio_pika
from aio_pika import Message, DeliveryMode
from aio_pika.abc import AbstractIncomingMessage

from app.config import settings

logger = logging.getLogger(__name__)


class TaskType(Enum):
    FILE = "file"
    URL = "url"
    TEXT = "text"


@dataclass
class TaskMessage:
    task_id: str
    task_type: TaskType
    source: str
    metadata: Dict[str, Any]
    created_at: str
    # For file tasks
    file_content: Optional[bytes] = None
    filename: Optional[str] = None
    # For URL tasks
    url: Optional[str] = None
    # For text tasks
    text_content: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # Convert enum to string
        data['task_type'] = self.task_type.value
        # Handle bytes serialization
        if self.file_content:
            import base64
            data['file_content'] = base64.b64encode(self.file_content).decode('utf-8')
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskMessage':
        # Convert string back to enum
        data['task_type'] = TaskType(data['task_type'])
        # Handle bytes deserialization
        if data.get('file_content'):
            import base64
            data['file_content'] = base64.b64decode(data['file_content'].encode('utf-8'))
        return cls(**data)


class RabbitMQHandler:
    def __init__(self):
        self.connection = None
        self.channel = None
        self.exchange = None
        self.queue = None
        self.is_connected = False

    async def connect(self):
        """Establish connection to RabbitMQ"""
        try:
            self.connection = await aio_pika.connect_robust(
                settings.RABBITMQ_URL,
                heartbeat=600,
                blocked_connection_timeout=300,
            )
            self.channel = await self.connection.channel()
            await self.channel.set_qos(prefetch_count=1)  # Process one message at a time
            
            # Declare exchange
            self.exchange = await self.channel.declare_exchange(
                settings.RABBITMQ_EXCHANGE_NAME,
                aio_pika.ExchangeType.DIRECT,
                durable=True
            )
            
            # Declare queue
            self.queue = await self.channel.declare_queue(
                settings.RABBITMQ_QUEUE_NAME,
                durable=True
            )
            
            # Bind queue to exchange
            await self.queue.bind(
                self.exchange,
                routing_key=settings.RABBITMQ_ROUTING_KEY
            )
            
            self.is_connected = True
            logger.info("Connected to RabbitMQ successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {str(e)}")
            raise

    async def disconnect(self):
        """Close RabbitMQ connection"""
        if self.connection and not self.connection.is_closed:
            await self.connection.close()
            self.is_connected = False
            logger.info("Disconnected from RabbitMQ")

    async def _ensure_connected(self):
        """Ensure we have a valid connection"""
        if not self.is_connected or self.connection.is_closed:
            await self.connect()

    async def enqueue_file(self, file_content: bytes, filename: str, source: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Enqueue a file processing task"""
        await self._ensure_connected()
        
        task_id = str(uuid.uuid4())
        if metadata is None:
            metadata = {}
        if 'filename' not in metadata:
            metadata['filename'] = filename
            
        task_message = TaskMessage(
            task_id=task_id,
            task_type=TaskType.FILE,
            source=source,
            metadata=metadata,
            created_at=datetime.now().isoformat(),
            file_content=file_content,
            filename=filename
        )
        
        await self._publish_message(task_message)
        logger.info(f"File task enqueued with ID: {task_id}")
        return task_id

    async def enqueue_url(self, url: str, source: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Enqueue a URL processing task"""
        await self._ensure_connected()
        
        task_id = str(uuid.uuid4())
        if metadata is None:
            metadata = {}
            
        task_message = TaskMessage(
            task_id=task_id,
            task_type=TaskType.URL,
            source=source,
            metadata=metadata,
            created_at=datetime.now().isoformat(),
            url=url
        )
        
        await self._publish_message(task_message)
        logger.info(f"URL task enqueued with ID: {task_id}")
        return task_id

    async def enqueue_text(self, text: str, source: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Enqueue a text processing task"""
        await self._ensure_connected()
        
        task_id = str(uuid.uuid4())
        if metadata is None:
            metadata = {}
            
        task_message = TaskMessage(
            task_id=task_id,
            task_type=TaskType.TEXT,
            source=source,
            metadata=metadata,
            created_at=datetime.now().isoformat(),
            text_content=text
        )
        
        await self._publish_message(task_message)
        logger.info(f"Text task enqueued with ID: {task_id}")
        return task_id

    async def _publish_message(self, task_message: TaskMessage):
        """Publish a message to RabbitMQ"""
        message_body = json.dumps(task_message.to_dict())
        
        message = Message(
            message_body.encode(),
            delivery_mode=DeliveryMode.PERSISTENT,  # Make message persistent
            message_id=task_message.task_id,
            timestamp=datetime.now(),
            headers={
                'task_type': task_message.task_type.value,
                'source': task_message.source,
            }
        )
        
        await self.exchange.publish(
            message,
            routing_key=settings.RABBITMQ_ROUTING_KEY
        )

    async def consume_messages(self, callback: Callable[[TaskMessage], None]):
        """Start consuming messages from the queue"""
        await self._ensure_connected()
        
        async def process_message(message: AbstractIncomingMessage):
            async with message.process():
                try:
                    # Parse the message
                    message_data = json.loads(message.body.decode())
                    task_message = TaskMessage.from_dict(message_data)
                    
                    logger.info(f"Processing task: {task_message.task_id} of type {task_message.task_type.value}")
                    
                    # Call the callback function
                    await callback(task_message)
                    
                    # Message is automatically acknowledged if no exception is raised
                    logger.info(f"Task {task_message.task_id} processed successfully")
                    
                except Exception as e:
                    logger.error(f"Error processing message: {str(e)}")
                    # Message will be requeued automatically due to exception
                    raise
        
        await self.queue.consume(process_message)
        logger.info("Started consuming messages from RabbitMQ queue")

    async def get_queue_info(self) -> Dict[str, Any]:
        """Get queue information"""
        await self._ensure_connected()
        
        queue_info = await self.queue.get_info()
        return {
            "queue_name": self.queue.name,
            "message_count": queue_info.messages,
            "consumer_count": queue_info.consumers,
            "is_connected": self.is_connected
        }

    async def purge_queue(self):
        """Purge all messages from the queue"""
        await self._ensure_connected()
        await self.queue.purge()
        logger.info("Queue purged successfully")


# Global instance
rabbitmq_handler = RabbitMQHandler()
