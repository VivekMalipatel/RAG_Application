import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, TYPE_CHECKING
import aio_pika
from aio_pika import Message, DeliveryMode

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
            
            self.queue = await self.channel.declare_queue(
                settings.RABBITMQ_QUEUE_NAME,
                durable=True,
                arguments={
                    "x-consumer-timeout": 36000000,
                }
            )
            
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
        if self.connection and not self.connection.is_closed:
            await self.connection.close()
            self.is_connected = False
            logger.info("Disconnected from RabbitMQ")

    async def _ensure_connected(self):
        if not self.is_connected or self.connection.is_closed:
            await self.connect()

    async def enqueue_task(self, task_message: TaskMessage) -> str:
        await self._ensure_connected()
        
        await self._publish_message(task_message)
        logger.info(f"Task enqueued with ID: {task_message.task_id}")
        return task_message.task_id

    async def _publish_message(self, task_message: TaskMessage):
        message_body = json.dumps(task_message.to_dict())
        
        message = Message(
            message_body.encode(),
            delivery_mode=DeliveryMode.PERSISTENT,
            message_id=task_message.task_id,
            timestamp=datetime.now(),
            headers={
                'task_type': task_message.task_type.value,
            }
        )
        
        await self.exchange.publish(
            message,
            routing_key=settings.RABBITMQ_ROUTING_KEY
        )

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
            # Get message with 10-hour timeout
            message = await self.queue.get(no_ack=False, fail=False, timeout=36000)
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
                
                await self._orchestrator.process(task_message)
                
                await message.ack()
                logger.info(f"Task {task_message.task_id} processed successfully and acknowledged")
                
            except Exception as e:
                logger.error(f"Error processing message: {str(e)}")
                await message.reject(requeue=False)
                logger.error(f"Task {task_message.task_id if 'task_message' in locals() else 'unknown'} rejected and sent to dead letter queue")
                raise
                
        except Exception:
            pass

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
        return {
            "queue_name": self.queue.name,
            "message_count": queue_info.messages,
            "consumer_count": queue_info.consumers,
            "is_connected": self.is_connected
        }

rabbitmq_handler = RabbitMQHandler()
