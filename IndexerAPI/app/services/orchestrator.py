import asyncio
import logging
import hashlib
from typing import Optional, Dict, Any

from core.queue.task_types import TaskMessage, TaskType
from core.model.model_handler import ModelHandler, get_global_model_handler
from services.vector_store import VectorStore, get_global_vector_store
from core.processors import FileProcessor, BaseProcessor

logger = logging.getLogger(__name__)

_global_orchestrator: Optional['Orchestrator'] = None


class Orchestrator:
    def __init__(self):
        self.model_handler = get_global_model_handler()
        self.processors = {}

    def register_processor(self, processor_type: str, processor: BaseProcessor):
        self.processors[processor_type] = processor
        logger.info(f"Registered {processor.__class__.__name__} for item type: {processor_type}")

    async def process(self, task_message: TaskMessage):
        try:
            if task_message.task_type == TaskType.FILE:
                await self._process_file_task(task_message)
            elif task_message.task_type == TaskType.URL:
                raise NotImplementedError("URL task processing not implemented")
            elif task_message.task_type == TaskType.TEXT:
                raise NotImplementedError("TEXT task processing not implemented")
            else:
                logger.error(f"Unknown task type: {task_message.task_type}")
                raise ValueError(f"Unknown task type: {task_message.task_type}")
                
        except Exception as e:
            logger.error(f"Error processing task {task_message.task_id}: {str(e)}")
            raise

    async def _process_file_task(self, task_message: TaskMessage):
        if "file" not in self.processors:
            raise ValueError("File processor not registered")
        processor: FileProcessor = self.processors["file"]
        try:
            logger.info(f"[Orchestrator] Processing file task {task_message.task_id}")
            result = await processor.process(task_message)
            logger.info(f"File task {task_message.task_id} completed successfully")
        except Exception as e:
            logger.error(f"Error processing file task {task_message.task_id}: {str(e)}")
            raise

    def cleanup(self):
        try:
            pass
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")


def get_global_orchestrator() -> Orchestrator:
    global _global_orchestrator
    if _global_orchestrator is None:
        _global_orchestrator = Orchestrator()
    return _global_orchestrator


async def cleanup_global_orchestrator():
    global _global_orchestrator
    if _global_orchestrator:
        _global_orchestrator.cleanup()
        _global_orchestrator = None
