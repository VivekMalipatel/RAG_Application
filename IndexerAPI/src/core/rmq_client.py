import asyncio
import json
import logging
from typing import Any, Dict, Optional
import aio_pika
from aio_pika import DeliveryMode, Message
from aio_pika.abc import AbstractRobustChannel, AbstractRobustConnection
from core.config import settings

logger = logging.getLogger(__name__)

_rmq_connection: Optional[AbstractRobustConnection] = None
_rmq_channel: Optional[AbstractRobustChannel] = None

def set_rmq_connection(connection: AbstractRobustConnection) -> None:
    global _rmq_connection
    _rmq_connection = connection

def set_rmq_channel(channel: AbstractRobustChannel) -> None:
    global _rmq_channel
    _rmq_channel = channel

async def connect_rabbitmq() -> AbstractRobustConnection:
    if not settings.RABBITMQ_URL:
        raise ValueError("RABBITMQ_URL is required")
    connection = await aio_pika.connect_robust(settings.RABBITMQ_URL, heartbeat=settings.RABBITMQ_HEARTBEAT, blocked_connection_timeout=settings.RABBITMQ_CONSUMER_TIMEOUT)
    return connection

async def declare_application_queues(channel: AbstractRobustChannel) -> None:
    if not settings.RABBITMQ_QUEUE_NAME:
        raise ValueError("RABBITMQ_QUEUE_NAME is required")

    main_queue_name = settings.RABBITMQ_QUEUE_NAME
    retry_queue_name = f"{main_queue_name}.retry"
    failed_queue_name = f"{main_queue_name}.failed"
    success_queue_name = f"{main_queue_name}.success"
    max_priority = 255
    retry_ttl_ms = max(settings.RABBITMQ_RETRY_DELAY_MS, 300000)
    failed_ttl_ms = max(settings.RABBITMQ_FAILED_TTL_MS, retry_ttl_ms)

    main_arguments = {
        "x-max-length": 1_000_000,
        "x-overflow": "drop-head",
        "x-consumer-timeout": settings.RABBITMQ_X_CONSUMER_TIMEOUT,
        "x-max-priority": max_priority,
        "x-dead-letter-exchange": "",
        "x-dead-letter-routing-key": retry_queue_name,
    }

    await channel.declare_queue(
        main_queue_name,
        durable=True,
        arguments=main_arguments,
    )

    retry_arguments = {
        "x-max-length": 1_000_000,
        "x-overflow": "drop-head",
        "x-message-ttl": retry_ttl_ms,
        "x-dead-letter-exchange": "",
        "x-dead-letter-routing-key": main_queue_name,
    }
    await channel.declare_queue(
        retry_queue_name,
        durable=True,
        arguments=retry_arguments,
    )

    failed_arguments = {
        "x-max-length": 1_000_000,
        "x-overflow": "drop-head",
        "x-message-ttl": failed_ttl_ms,
        "x-dead-letter-exchange": "",
        "x-dead-letter-routing-key": main_queue_name,
    }
    await channel.declare_queue(
        failed_queue_name,
        durable=True,
        arguments=failed_arguments,
    )

    success_arguments = {"x-max-length": 1_000_000, "x-overflow": "drop-head"}
    await channel.declare_queue(
        success_queue_name,
        durable=True,
        arguments=success_arguments,
    )

async def ensure_rmq_connection() -> None:
    global _rmq_connection, _rmq_channel
    if _rmq_connection and not _rmq_connection.is_closed and _rmq_channel and not _rmq_channel.is_closed:
        return
    for attempt in range(3):
        try:
            connection = await connect_rabbitmq()
            channel = await connection.channel()
            await declare_application_queues(channel)
            set_rmq_connection(connection)
            set_rmq_channel(channel)
            return
        except Exception as exc:
            logger.error(f"RabbitMQ connection attempt {attempt + 1} failed: {exc}")
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
    raise RuntimeError("Unable to establish RabbitMQ connection")

def get_rmq_connection() -> Optional[AbstractRobustConnection]:
    return _rmq_connection

def get_rmq_channel() -> Optional[AbstractRobustChannel]:
    return _rmq_channel

async def publish_message(payload: Dict[str, Any], routing_key: Optional[str] = None, headers: Optional[Dict[str, Any]] = None, priority: Optional[int] = None, exchange_name: Optional[str] = None) -> None:
    await ensure_rmq_connection()
    if not _rmq_channel:
        raise RuntimeError("RabbitMQ channel is not available")
    target_routing_key = routing_key or settings.RABBITMQ_QUEUE_NAME
    if not target_routing_key:
        raise ValueError("routing_key is required")
    requested_priority = priority if priority is not None else 0
    message_priority = max(0, min(requested_priority, 255))
    message = Message(json.dumps(payload).encode(), delivery_mode=DeliveryMode.PERSISTENT, headers=headers or {}, priority=message_priority)
    if exchange_name:
        exchange = await _rmq_channel.get_exchange(exchange_name)
        await exchange.publish(message, routing_key=target_routing_key)
    else:
        await _rmq_channel.default_exchange.publish(message, routing_key=target_routing_key)

async def close_rabbitmq() -> None:
    global _rmq_connection, _rmq_channel
    if _rmq_channel and not _rmq_channel.is_closed:
        await _rmq_channel.close()
    if _rmq_connection and not _rmq_connection.is_closed:
        await _rmq_connection.close()
    _rmq_channel = None
    _rmq_connection = None
