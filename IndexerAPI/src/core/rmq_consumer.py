import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional
from aio_pika import IncomingMessage, Message
from core.config import settings
from core.logger_setup import logger
from core.rmq_client import ensure_rmq_connection, get_rmq_connection

class BaseConsumer(ABC):
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.queue = None
        self.consumer_tag: Optional[str] = None
        self.is_running = False
        self.channel = None

    @abstractmethod
    async def process_message(self, message: IncomingMessage) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_queue_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_dlq_name(self) -> str:
        raise NotImplementedError

    async def setup_channel(self) -> None:
        await ensure_rmq_connection()
        connection = get_rmq_connection()
        if not connection:
            raise ValueError("RabbitMQ channel not available")
        self.channel = await connection.channel()
        await self.channel.set_qos(prefetch_count=max(1, settings.RABBITMQ_PREFETCH_COUNT))

    async def setup_queue(self) -> None:
        queue_name = self.get_queue_name()
        if not queue_name:
            raise ValueError(f"{self.service_name}: Queue name not configured")
        self.queue = await self.channel.declare_queue(queue_name, passive=True)

    async def _message_wrapper(self, message: IncomingMessage) -> None:
        try:
            await self.process_message(message)
            logger.debug(f"{self.service_name}: Message processed successfully")
        except Exception as exc:
            logger.error(f"{self.service_name}: Error processing message: {exc}")
            dlq_name = self.get_dlq_name()
            if dlq_name:
                try:
                    dlq_message = Message(
                        message.body,
                        headers={
                            "x-original-queue": self.get_queue_name(),
                            "x-error": str(exc),
                            "x-failed-at": datetime.now().isoformat(),
                            "x-service": self.service_name,
                        },
                    )
                    await self.channel.default_exchange.publish(dlq_message, routing_key=dlq_name)
                    logger.info(f"{self.service_name}: Message sent to DLQ: {dlq_name}")
                except Exception as dlq_error:
                    logger.error(f"{self.service_name}: Failed to publish to DLQ: {dlq_error}")
            else:
                logger.warning(f"{self.service_name}: No DLQ configured, message discarded")

    async def start_consuming(self) -> None:
        await self.setup_channel()
        await self.setup_queue()
        self.consumer_tag = await self.queue.consume(self._message_wrapper, no_ack=False)
        self.is_running = True
        while self.is_running:
            await asyncio.sleep(1)

    async def stop_consuming(self) -> None:
        self.is_running = False
        if self.queue and self.consumer_tag:
            await self.queue.cancel(self.consumer_tag)

    async def run(self) -> None:
        try:
            await self.start_consuming()
        except KeyboardInterrupt:
            pass
        finally:
            await self.stop_consuming()
