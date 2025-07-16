from core.processors.base_processor import BaseProcessor
from core.processors.file_processor import FileProcessor

__all__ = ["BaseProcessor", "FileProcessor", "register_processors"]

def register_processors(orchestrator):
    processors = [
        ("file", FileProcessor()),
    ]
    
    for item_type, processor in processors:
        orchestrator.register_processor(item_type, processor)
    
    return orchestrator