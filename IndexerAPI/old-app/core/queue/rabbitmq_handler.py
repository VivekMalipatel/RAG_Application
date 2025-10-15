import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, TYPE_CHECKING
import aio_pika
from aio_pika import Message, DeliveryMode
from aio_pika.abc import AbstractIncomingMessage

from core.queue.task_types import TaskMessage, TaskType
from config import settings

if TYPE_CHECKING:
    from services.orchestrator import Orchestrator

logger = logging.getLogger(__name__)


class RabbitMQHandler:
    def __init__(self):
        self.connection = None
        self.channel = None
        self.exchange = None
        self.queue = None
        self.dlx_exchange = None
        self.dlq = None
        self.success_exchange = None
        self.success_queue = None
        self.is_connected = False
        self.is_consuming = False
        self._consumer_task = None
        self._orchestrator: 'Orchestrator' = None

    async def connect(self):
        try:
            self.connection = await aio_pika.connect_robust(
                settings.RABBITMQ_URL,
                heartbeat=settings.RABBITMQ_HEARTBEAT,
                blocked_connection_timeout=settings.RABBITMQ_CONSUMER_TIMEOUT,
            )
            self.channel = await self.connection.channel()
            await self.channel.set_qos(prefetch_count=settings.MAX_DEQUEUE_CONCURRENCY)
            
            self.exchange = await self.channel.declare_exchange(
                settings.RABBITMQ_EXCHANGE_NAME,
                aio_pika.ExchangeType.DIRECT,
                durable=True
            )
            
            await self._setup_dead_letter_queue()
            await self._setup_success_queue()
            
            self.queue = await self.channel.declare_queue(
                settings.RABBITMQ_QUEUE_NAME,
                durable=True,
                arguments={
                    "x-consumer-timeout": settings.RABBITMQ_X_CONSUMER_TIMEOUT,
                    "x-dead-letter-exchange": f"{settings.RABBITMQ_EXCHANGE_NAME}.dlx",
                    "x-dead-letter-routing-key": f"{settings.RABBITMQ_ROUTING_KEY}.failed"
                }
            )
            
            await self.queue.bind(
                self.exchange,
                routing_key=settings.RABBITMQ_ROUTING_KEY
            )
            
            self.is_connected = True
            logger.info("Connected to RabbitMQ with extended timeout configuration")
            logger.info(f"Timeout configuration - Heartbeat: {settings.RABBITMQ_HEARTBEAT}s, Connection timeout: {settings.RABBITMQ_CONSUMER_TIMEOUT}s, Queue timeout: {settings.RABBITMQ_X_CONSUMER_TIMEOUT}ms, Message get timeout: {settings.RABBITMQ_MESSAGE_GET_TIMEOUT}s")
            
        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {str(e)}")
            raise

    async def disconnect(self):
        if self.connection and not self.connection.is_closed:
            await self.connection.close()
            self.is_connected = False
            logger.info("Disconnected from RabbitMQ")

    async def _ensure_connected(self):
        if not self.is_connected or self.connection.is_closed:
            await self.connect()

    async def _setup_dead_letter_queue(self):
        self.dlx_exchange = await self.channel.declare_exchange(
            f"{settings.RABBITMQ_EXCHANGE_NAME}.dlx",
            aio_pika.ExchangeType.DIRECT,
            durable=True
        )
        
        self.dlq = await self.channel.declare_queue(
            f"{settings.RABBITMQ_QUEUE_NAME}.failed",
            durable=True
        )
        
        await self.dlq.bind(
            self.dlx_exchange,
            routing_key=f"{settings.RABBITMQ_ROUTING_KEY}.failed"
        )
        
        logger.info("Dead letter queue setup completed")

    async def _setup_success_queue(self):
        self.success_exchange = await self.channel.declare_exchange(
            f"{settings.RABBITMQ_EXCHANGE_NAME}.success",
            aio_pika.ExchangeType.DIRECT,
            durable=True
        )
        
        self.success_queue = await self.channel.declare_queue(
            f"{settings.RABBITMQ_QUEUE_NAME}.success",
            durable=True
        )
        
        await self.success_queue.bind(
            self.success_exchange,
            routing_key=f"{settings.RABBITMQ_ROUTING_KEY}.success"
        )
        
        logger.info("Success queue setup completed")

    async def enqueue_task(self, task_message: TaskMessage) -> str:
        await self._ensure_connected()
        
        await self._publish_message(task_message)
        logger.info(f"Task enqueued with ID: {task_message.task_id}")
        return task_message.task_id

    async def _publish_message(self, task_message: TaskMessage, retry_count: int = 0):
        message_body = json.dumps(task_message.to_dict())
        
        message = Message(
            message_body.encode(),
            delivery_mode=DeliveryMode.PERSISTENT,
            message_id=task_message.task_id,
            timestamp=datetime.now(),
            headers={
                'task_type': task_message.task_type.value,
                'retry_count': retry_count,
                'max_retries': 3,
            }
        )
        
        await self.exchange.publish(
            message,
            routing_key=settings.RABBITMQ_ROUTING_KEY
        )

    async def _send_to_dead_letter_queue(self, task_message: TaskMessage, error_message: str):
        try:
            await self._ensure_connected()
            
            dlq_message_data = {
                **task_message.to_dict(),
                "error": error_message,
                "failed_at": datetime.now().isoformat(),
                "original_queue": settings.RABBITMQ_QUEUE_NAME
            }
            
            message_body = json.dumps(dlq_message_data)
            
            message = Message(
                message_body.encode(),
                delivery_mode=DeliveryMode.PERSISTENT,
                message_id=task_message.task_id,
                timestamp=datetime.now(),
                headers={
                    'task_type': task_message.task_type.value,
                    'failure_reason': error_message,
                    'original_task_id': task_message.task_id
                }
            )
            
            await self.dlx_exchange.publish(
                message,
                routing_key=f"{settings.RABBITMQ_ROUTING_KEY}.failed"
            )
            
            logger.info(f"Task {task_message.task_id} sent to dead letter queue due to: {error_message}")
            
        except Exception as e:
            logger.error(f"Failed to send task {task_message.task_id} to dead letter queue: {str(e)}")

    async def _send_to_success_queue(self, task_message: TaskMessage, processing_time: float):
        try:
            await self._ensure_connected()
            
            success_message_data = {
                **task_message.to_dict(),
                "completed_at": datetime.now().isoformat(),
                "processing_time_seconds": processing_time,
                "status": "completed",
                "original_queue": settings.RABBITMQ_QUEUE_NAME
            }
            
            message_body = json.dumps(success_message_data)
            
            message = Message(
                message_body.encode(),
                delivery_mode=DeliveryMode.PERSISTENT,
                message_id=f"success_{task_message.task_id}",
                timestamp=datetime.now(),
                headers={
                    'task_type': task_message.task_type.value,
                    'status': 'completed',
                    'original_task_id': task_message.task_id,
                    'processing_time': str(processing_time)
                }
            )
            
            await self.success_exchange.publish(
                message,
                routing_key=f"{settings.RABBITMQ_ROUTING_KEY}.success"
            )
            
            logger.info(f"Task {task_message.task_id} sent to success queue after {processing_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Failed to send task {task_message.task_id} to success queue: {str(e)}")

    async def start_consuming(self, orchestrator):
        if self.is_consuming:
            logger.warning("Consumer is already running")
            return

        await self._ensure_connected()
        self.is_consuming = True
        self._orchestrator = orchestrator
        logger.info(f"Starting RabbitMQ consumer with max concurrency: {settings.MAX_DEQUEUE_CONCURRENCY}")
        self._consumer_task = asyncio.create_task(self._consume_loop())

    async def _consume_loop(self):
        tasks = set()
        logger.info("Consumer loop started")
        while self.is_consuming:
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
            await self._ensure_connected()
            message = await self.queue.get(
                no_ack=False, 
                fail=False, 
                timeout=settings.RABBITMQ_MESSAGE_GET_TIMEOUT
            )
            if message is None:
                return None
            return message
        except Exception as e:
            logger.error(f"Error getting message from queue: {str(e)}")
            return None

    async def _handle_message(self, message: AbstractIncomingMessage):
        task_message = None
        processing_start_time = time.time()
        
        try:
            message_data = json.loads(message.body.decode())
            task_message = TaskMessage.from_dict(message_data)
            
            retry_count = message.headers.get('retry_count', 0) if message.headers else 0
            max_retries = message.headers.get('max_retries', 3) if message.headers else 3
            
            logger.info(f"Starting task: {task_message.task_id} of type {task_message.task_type.value} (attempt {retry_count + 1})")
            
            await self._orchestrator.process(task_message)
            
            await message.ack()
            
            processing_time = time.time() - processing_start_time
            logger.info(f"Task {task_message.task_id} completed successfully in {processing_time:.2f} seconds and acknowledged")
            
            await self._send_to_success_queue(task_message, processing_time)
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid message format: {str(e)}")
            await message.reject(requeue=False)
            
        except Exception as e:
            processing_time = time.time() - processing_start_time
            logger.error(f"Task {task_message.task_id if task_message else 'unknown'} failed after {processing_time:.2f} seconds: {str(e)}")
            
            retry_count = message.headers.get('retry_count', 0) if message.headers else 0
            max_retries = message.headers.get('max_retries', 3) if message.headers else 3
            
            if self._should_retry(e, task_message) and retry_count < max_retries:
                await message.reject(requeue=False)
                if task_message:
                    await self._requeue_with_retry(task_message, retry_count + 1)
            else:
                await message.reject(requeue=False)
                if task_message:
                    await self._send_to_dead_letter_queue(task_message, str(e))
                logger.error(f"Task {task_message.task_id if task_message else 'unknown'} rejected and sent to dead letter queue")

    def _should_retry(self, error: Exception, task_message: TaskMessage) -> bool:
        non_retryable_errors = (
            json.JSONDecodeError,
            ValueError,
            TypeError,
            KeyError,
        )
        
        if isinstance(error, non_retryable_errors):
            return False
        
        retryable_errors = (
            ConnectionError,
            TimeoutError,
            aio_pika.exceptions.ConnectionClosed,
            aio_pika.exceptions.ChannelClosed,
        )
        
        return isinstance(error, retryable_errors)

    async def _requeue_with_retry(self, task_message: TaskMessage, retry_count: int):
        await self._publish_message(task_message, retry_count)
        logger.info(f"Task {task_message.task_id} requeued for retry attempt {retry_count + 1}")

    async def stop_consuming(self):
        if not self.is_consuming:
            return
        self.is_consuming = False
        if self._consumer_task:
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except asyncio.CancelledError:
                logger.info("Consumer task cancelled")
        logger.info("RabbitMQ consumer stopped")

    async def get_queue_info(self) -> Dict[str, Any]:
        await self._ensure_connected()
        
        queue_info = await self.queue.get_info()
        dlq_info = await self.dlq.get_info()
        success_info = await self.success_queue.get_info()
        
        return {
            "main_queue": {
                "queue_name": self.queue.name,
                "message_count": queue_info.messages,
                "consumer_count": queue_info.consumers,
            },
            "dead_letter_queue": {
                "queue_name": self.dlq.name,
                "message_count": dlq_info.messages,
                "consumer_count": dlq_info.consumers,
            },
            "success_queue": {
                "queue_name": self.success_queue.name,
                "message_count": success_info.messages,
                "consumer_count": success_info.consumers,
            },
            "is_connected": self.is_connected
        }

rabbitmq_handler = RabbitMQHandler()
