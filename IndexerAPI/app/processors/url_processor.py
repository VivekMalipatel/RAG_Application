import logging
import re
from typing import Dict, Any, List, Optional
from datetime import datetime
import urllib.parse

from app.processors.base_processor import BaseProcessor
from app.core.markitdown.markdown_handler import MarkDown

logger = logging.getLogger(__name__)

class URLProcessor(BaseProcessor):
    
    def __init__(self):
        self.markdown = MarkDown()
        logger.info("URLProcessor initialized")
    
    def is_youtube_url(self, url: str) -> bool:
        youtube_patterns = [
            r'^(https?://)?(www\.)?(youtube\.com|youtu\.?be)/.+$',
        ]
        
        for pattern in youtube_patterns:
            if re.match(pattern, url):
                return True
        return False
    
    async def process(self, data: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        logger.info("Processing URL")
        
        if not data or not isinstance(data, str):
            error_msg = f"Invalid data type for URL processing: {type(data)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        try:
            is_youtube = self.is_youtube_url(data)
            
            if is_youtube:
                result = await self.process_youtube_url(data)
            else:
                result = {
                    "data": [self.markdown.convert_url(data)],
                }
            
            result["metadata"] = metadata or {}
            logger.info(f"Successfully processed URL: {data}")
            return result
            
        except Exception as e:
            error_msg = f"Error processing URL: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise
            
    async def process_youtube_url(self, url: str) -> Dict[str, Any]:
        try:
            markdown_text = self.markdown.convert_url(url)
            return {
                "data": [markdown_text]
            }
        except Exception as e:
            logger.error(f"Error processing YouTube URL: {e}")
            raise