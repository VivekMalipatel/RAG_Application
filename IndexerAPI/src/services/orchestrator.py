import logging
from typing import Dict, Optional
from core.queue.task_types import TaskMessage, TaskType
from core.model.model_handler import get_global_model_handler
from core.processors import BaseProcessor

logger = logging.getLogger(__name__)

_global_orchestrator: Optional["Orchestrator"] = None

class Orchestrator:
    def __init__(self):
        self.model_handler = get_global_model_handler()
        self.processors: Dict[TaskType, BaseProcessor] = {}

    def register_processor(self, task_type: TaskType, processor: BaseProcessor):
        self.processors[task_type] = processor
        logger.info(f"Registered {processor.__class__.__name__} for task type: {task_type.value}")

    async def process(self, task_message: TaskMessage):
        try:
            processor = self.processors.get(task_message.task_type)
            if not processor:
                if task_message.task_type == TaskType.URL:
                    raise NotImplementedError("URL task processing not implemented")
                if task_message.task_type == TaskType.TEXT:
                    raise NotImplementedError("TEXT task processing not implemented")
                logger.error(f"Unknown task type: {task_message.task_type}")
                raise ValueError(f"Unknown task type: {task_message.task_type}")
            logger.info(f"[Orchestrator] Dispatching task {task_message.task_id} for {task_message.task_type.value}")
            await processor.process(task_message)
            logger.info(f"Task {task_message.task_id} for {task_message.task_type.value} completed")
        except Exception as exc:
            logger.error(f"Error processing task {task_message.task_id}: {exc}")
            raise

    def cleanup(self):
        try:
            pass
        except Exception as exc:
            logger.error(f"Error saving vector store: {exc}")

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
