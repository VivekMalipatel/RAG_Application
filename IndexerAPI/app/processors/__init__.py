from app.processors.base_processor import BaseProcessor
from app.processors.raw_text_processor import RawTextProcessor
from app.processors.url_processor import URLProcessor
from app.processors.file_processor import FileProcessor

__all__ = ["BaseProcessor", "RawTextProcessor", "URLProcessor", "FileProcessor", "register_processors"]

def register_processors(queue_consumer):

    processors = [
        ("text", RawTextProcessor()),
        ("url", URLProcessor()),
        ("file", FileProcessor()),
    ]
    
    for item_type, processor in processors:
        queue_consumer.register_processor(item_type, processor)
    
    return queue_consumer