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
            error_msg = f"Invalid data type for raw text processing: {type(data)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        try:
            markdown_text = self.markdown.convert_text(data)
            logger.debug(f"Converted raw text to markdown (length: {len(markdown_text)} chars)")
            
            return {
                "data": [markdown_text],
                "metadata": metadata or {}
            }
        except Exception as e:
            error_msg = f"Error processing raw text: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise