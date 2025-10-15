from core.processors.base_processor import BaseProcessor
from core.processors.file_processor import FileProcessor
from core.processors._unstructured_processor import UnstructuredProcessor
from core.processors._structured_processor import StructuredProcessor
from core.processors._direct_processor import DirectProcessor
from core.queue.task_types import TaskType

__all__ = [
    "BaseProcessor",
    "FileProcessor",
    "register_processors",
]

def register_processors(orchestrator):
    orchestrator.register_processor(TaskType.FILE, FileProcessor())
    orchestrator.register_processor(TaskType.UNSTRUCTURED_PAGE, UnstructuredProcessor())
    orchestrator.register_processor(TaskType.STRUCTURED_CHUNK, StructuredProcessor())
    orchestrator.register_processor(TaskType.DIRECT_CHUNK, DirectProcessor())
    return orchestrator
