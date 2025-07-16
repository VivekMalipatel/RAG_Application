from core.processors.base_processor import BaseProcessor
from core.processors.raw_text_processor import RawTextProcessor
from core.processors.url_processor import URLProcessor
from core.processors.file_processor import FileProcessor

__all__ = ["BaseProcessor", "RawTextProcessor", "URLProcessor", "FileProcessor", "register_processors"]

def register_processors(orchestrator):
    processors = [
        ("text", RawTextProcessor()),
        ("url", URLProcessor()),
        ("file", FileProcessor()),
    ]
    
    for item_type, processor in processors:
        orchestrator.register_processor(item_type, processor)
    
    return orchestrator