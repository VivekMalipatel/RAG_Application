import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from app.processors.base_processor import BaseProcessor
from app.core.markitdown.markdown_handler import MarkDown

logger = logging.getLogger(__name__)

class RawTextProcessor(BaseProcessor):
    
    def __init__(self):
        self.markdown = MarkDown()
        logger.info("RawTextProcessor initialized")
    
    async def process(self, data: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        logger.info("Processing raw text")
        
        if not data or not isinstance(data, str):
            error_msg = f"Invalid data type for text processing: {type(data)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        try:
            result = {
                "data": [self.markdown.convert_text(data)],
                "metadata": metadata or {}
            }
            
            logger.info(f"Successfully processed raw text of {len(data)} characters")
            return result
            
        except Exception as e:
            error_msg = f"Error processing raw text: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise