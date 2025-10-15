import asyncio
import json
import time
import traceback
from datetime import datetime
from typing import Any, Dict, Optional

from aio_pika.abc import AbstractIncomingMessage

from core.config import settings
from core.logger_setup import logger
from core.queue.task_types import TaskMessage, TaskType
from core.rmq_client import ensure_rmq_connection, get_rmq_channel, publish_message
from core.rmq_consumer import BaseConsumer

def _success_queue_name() -> str:
    queue_name = settings.RABBITMQ_QUEUE_NAME
    return f"{queue_name}.success" if queue_name else ""


def _retry_queue_name() -> str:
    queue_name = settings.RABBITMQ_QUEUE_NAME
    return f"{queue_name}.retry" if queue_name else ""


def _failed_queue_name() -> str:
    queue_name = settings.RABBITMQ_QUEUE_NAME
    return f"{queue_name}.failed" if queue_name else ""


def _max_priority() -> int:
    return max(1, settings.RABBITMQ_MAX_PRIORITY)


def _calculate_task_priority(task_message: TaskMessage) -> int:
    max_priority = _max_priority()
    task_type = task_message.task_type
    if task_type in {TaskType.FILE, TaskType.URL, TaskType.TEXT}:
        return max_priority
    if task_type == TaskType.UNSTRUCTURED_PAGE:
        page_number = task_message.payload.get("page_number")
        if isinstance(page_number, int) and page_number > 0:
            highest_page_priority = max_priority - 1 if max_priority > 1 else 0
            priority_offset = page_number - 1
            calculated_priority = highest_page_priority - priority_offset
            return max(0, calculated_priority)
        return max(0, max_priority - 1)
    return 1

async def _send_to_success_queue(task_message: TaskMessage, processing_time: float) -> None:
    payload = {**task_message.to_dict(), "completed_at": datetime.now().isoformat(), "processing_time_seconds": processing_time, "status": "completed", "original_queue": settings.RABBITMQ_QUEUE_NAME}
    await publish_message(payload, routing_key=_success_queue_name())


async def _send_to_failed_queue(task_message: Optional[TaskMessage], error_message: str, stack_trace: str, attempt: int, original_body: bytes) -> None:
    payload: Dict[str, Any] = {
        "error": error_message,
        "stack_trace": stack_trace,
        "attempt": attempt,
        "failed_at": datetime.now().isoformat(),
    }
    if task_message:
        payload["task"] = task_message.to_dict()
    else:
        try:
            payload["raw_body"] = json.loads(original_body.decode())
        except Exception:
            payload["raw_body"] = original_body.decode(errors="replace")
    await publish_message(payload, routing_key=_failed_queue_name())

class TaskQueueConsumer(BaseConsumer):
    def __init__(self, orchestrator):
        super().__init__("TaskQueueConsumer")
        self._orchestrator = orchestrator
        self._max_retries = max(1, settings.RABBITMQ_MAX_RETRIES)

    def get_queue_name(self) -> str:
        return settings.RABBITMQ_QUEUE_NAME or ""

    def get_dlq_name(self) -> str:
        if not settings.RABBITMQ_QUEUE_NAME:
            return ""
        return _failed_queue_name()

    def get_retry_queue_name(self) -> str:
        return _retry_queue_name()

    def _delivery_attempt(self, message: AbstractIncomingMessage) -> int:
        death_header = message.headers.get("x-death") if message.headers else None
        if not death_header:
            return 1
        if isinstance(death_header, list):
            total = 1
            for entry in death_header:
                total += int(entry.get("count", 0))
            return total
        return 1

    def _calculate_retry_decision(self, attempt: int) -> bool:
        return attempt < self._max_retries

    async def process_message(self, message: AbstractIncomingMessage) -> None:
        start_time = time.time()
        task_message: Optional[TaskMessage] = None
        attempt = self._delivery_attempt(message)
        try:
            data = json.loads(message.body.decode())
            task_message = TaskMessage.from_dict(data)
        except json.JSONDecodeError as exc:
            stack_trace = traceback.format_exc()
            logger.error(f"Invalid message format: {exc}")
            await _send_to_failed_queue(None, f"Invalid message: {exc}", stack_trace, attempt, message.body)
            await message.ack()
            return

        try:
            logger.info(f"Starting task: {task_message.task_id} (attempt {attempt})")
            await self._orchestrator.process(task_message)
        except Exception as exc:
            elapsed = time.time() - start_time
            stack_trace = traceback.format_exc()
            logger.error(
                f"Task {task_message.task_id} failed in {elapsed:.2f}s on attempt {attempt}: {exc}"
            )
            if self._calculate_retry_decision(attempt):
                await message.reject(requeue=False)
                logger.info(
                    f"Task {task_message.task_id} scheduled for retry via {_retry_queue_name()} (attempt {attempt})"
                )
            else:
                await _send_to_failed_queue(task_message, str(exc), stack_trace, attempt, message.body)
                await message.ack()
            return

        elapsed = time.time() - start_time
        await message.ack()
        logger.info(f"Task {task_message.task_id} completed in {elapsed:.2f}s")
        await _send_to_success_queue(task_message, elapsed)

class RabbitMQHandler:
    def __init__(self):
        self._consumer: Optional[TaskQueueConsumer] = None
        self._consumer_task: Optional[asyncio.Task] = None

    async def enqueue_task(self, task_message: TaskMessage) -> str:
        priority = _calculate_task_priority(task_message)
        headers = {"task_type": task_message.task_type.value, "priority": priority}
        await publish_message(task_message.to_dict(), headers=headers, priority=priority)
        return task_message.task_id

    async def start_consuming(self, orchestrator) -> None:
        if self._consumer_task and not self._consumer_task.done():
            return
        self._consumer = TaskQueueConsumer(orchestrator)
        async def _run_consumer():
            await self._consumer.run()
        self._consumer_task = asyncio.create_task(_run_consumer())
        await self._consumer_task

    async def stop_consuming(self) -> None:
        if self._consumer:
            await self._consumer.stop_consuming()
        if self._consumer_task:
            await self._consumer_task
        self._consumer = None
        self._consumer_task = None

    async def get_queue_info(self) -> Dict[str, Any]:
        await ensure_rmq_connection()
        channel = get_rmq_channel()
        if not channel:
            raise RuntimeError("RabbitMQ channel is not available")
        queue = await channel.declare_queue(settings.RABBITMQ_QUEUE_NAME, passive=True)
        dlq = await channel.declare_queue(_failed_queue_name(), passive=True)
        retry_queue = await channel.declare_queue(_retry_queue_name(), passive=True)
        success_queue = await channel.declare_queue(f"{settings.RABBITMQ_QUEUE_NAME}.success", passive=True)
        return {
            "main_queue": {"queue_name": queue.name, "message_count": queue.declaration_result.message_count, "consumer_count": queue.declaration_result.consumer_count},
            "dead_letter_queue": {"queue_name": dlq.name, "message_count": dlq.declaration_result.message_count, "consumer_count": dlq.declaration_result.consumer_count},
            "retry_queue": {"queue_name": retry_queue.name, "message_count": retry_queue.declaration_result.message_count, "consumer_count": retry_queue.declaration_result.consumer_count},
            "success_queue": {"queue_name": success_queue.name, "message_count": success_queue.declaration_result.message_count, "consumer_count": success_queue.declaration_result.consumer_count}
        }

rabbitmq_handler = RabbitMQHandler()
