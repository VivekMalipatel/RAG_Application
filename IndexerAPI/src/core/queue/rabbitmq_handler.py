import asyncio
import json
import time
import traceback
from datetime import datetime
from typing import Any, Dict, Optional

from aio_pika.abc import AbstractIncomingMessage
from botocore.exceptions import ClientError

try:
    from pandas.errors import EmptyDataError
except Exception:
    EmptyDataError = None

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


def _should_discard_error(exc: Exception) -> bool:
    seen = set()
    stack = [exc]
    while stack:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, ClientError):
            error_info = getattr(current, "response", None)
            if not isinstance(error_info, dict):
                error_info = {}
            error_details = error_info.get("Error", {})
            error_code = error_details.get("Code") if isinstance(error_details, dict) else None
            if error_code == "NoSuchKey":
                return True
        if EmptyDataError is not None and isinstance(current, EmptyDataError):
            return True
        for nested in (getattr(current, "__cause__", None), getattr(current, "__context__", None)):
            if nested is not None:
                stack.append(nested)
    message = str(exc)
    if "NoSuchKey" in message:
        return True
    if "No columns to parse from file" in message:
        return True
    return False


def _max_priority() -> int:
    return 255


def _calculate_task_priority(task_message: TaskMessage) -> int:
    max_priority = _max_priority()
    task_type = task_message.task_type
    payload = task_message.payload or {}
    if task_type in {TaskType.FILE, TaskType.URL, TaskType.TEXT}:
        return max_priority
    if task_type == TaskType.UNSTRUCTURED_PAGE:
        page_number = payload.get("page_number")
        base_priority = max_priority - 50 if max_priority > 50 else max_priority
        return _priority_from_index(base_priority, page_number)
    if task_type == TaskType.STRUCTURED_CHUNK:
        chunk_index = payload.get("chunk_index")
        base_priority = max_priority - 5 if max_priority > 5 else max_priority
        return _priority_from_index(base_priority, chunk_index)
    if task_type == TaskType.DIRECT_CHUNK:
        chunk_index = payload.get("chunk_index")
        base_priority = max_priority - 25 if max_priority > 25 else max_priority
        return _priority_from_index(base_priority, chunk_index)
    return 1


def _priority_from_index(base_priority: int, index: Any) -> int:
    if isinstance(index, int) and index > 0:
        adjusted = base_priority - (index - 1)
        return max(0, adjusted)
    return max(0, base_priority)

async def _send_to_success_queue(task_message: TaskMessage, processing_time: float) -> None:
    payload = {**task_message.to_dict(), "completed_at": datetime.now().isoformat(), "processing_time_seconds": processing_time, "status": "completed", "original_queue": settings.RABBITMQ_QUEUE_NAME}
    await publish_message(payload, routing_key=_success_queue_name())


async def _send_to_failed_queue(task_message: Optional[TaskMessage], error_message: str, stack_trace: str, attempt: int, original_body: bytes) -> None:
    headers: Dict[str, Any] = {
        "failure_reason": error_message,
        "failure_attempt": attempt,
        "failure_timestamp": datetime.now().isoformat(),
    }
    if task_message is not None:
        payload = task_message.to_dict()
        priority = _calculate_task_priority(task_message)
        headers["task_type"] = task_message.task_type.value
        await publish_message(payload, routing_key=_failed_queue_name(), headers=headers, priority=priority)
        return

    metadata_payload: Dict[str, Any] = {
        "error": error_message,
        "stack_trace": stack_trace,
        "attempt": attempt,
        "failed_at": datetime.now().isoformat(),
    }
    try:
        metadata_payload["raw_body"] = json.loads(original_body.decode())
    except Exception:
        metadata_payload["raw_body"] = original_body.decode(errors="replace")
    logger.warning("Dropping invalid task payload after failure: %s", metadata_payload)

async def _send_to_retry_queue(task_message: TaskMessage, error_message: str, attempt: int, original_headers: Optional[Dict[str, Any]]) -> None:
    headers: Dict[str, Any] = dict(original_headers or {})
    headers.update({
        "failure_reason": error_message,
        "failure_attempt": attempt,
        "failure_timestamp": datetime.now().isoformat(),
        "original_queue": settings.RABBITMQ_QUEUE_NAME,
        "task_type": task_message.task_type.value,
    })
    payload = task_message.to_dict()
    priority = _calculate_task_priority(task_message)
    await publish_message(payload, routing_key=_retry_queue_name(), headers=headers, priority=priority)

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
            if _should_discard_error(exc):
                await _send_to_failed_queue(task_message, str(exc), stack_trace, attempt, message.body)
                await message.ack()
                logger.info(f"Task {task_message.task_id} discarded due to non-retryable error")
                return
            if self._calculate_retry_decision(attempt):
                try:
                    await _send_to_retry_queue(task_message, str(exc), attempt, message.headers)
                    await message.ack()
                    logger.info(
                        f"Task {task_message.task_id} forwarded to {_retry_queue_name()} (attempt {attempt})"
                    )
                except Exception as retry_error:
                    logger.error(f"Failed to forward task {task_message.task_id} to retry queue: {retry_error}")
                    await _send_to_failed_queue(task_message, str(exc), stack_trace, attempt, message.body)
                    await message.ack()
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
