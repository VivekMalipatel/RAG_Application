from app.processors.base_processor import BaseProcessor
from app.processors.raw_text_processor import RawTextProcessor

__all__ = ["BaseProcessor", "RawTextProcessor", "register_processors"]

def register_processors(queue_consumer):

    processors = [
        ("text", RawTextProcessor()),
    ]
    
    for item_type, processor in processors:
        queue_consumer.register_processor(item_type, processor)
    
    return queue_consumer