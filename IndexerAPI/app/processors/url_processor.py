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
        """Check if the URL is a YouTube URL."""
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
            parsed_url = urllib.parse.urlparse(data)
            if not all([parsed_url.scheme, parsed_url.netloc]):
                raise ValueError(f"Invalid URL format: {data}")
            
            if self.is_youtube_url(data):
                logger.info(f"Processing YouTube URL: {data}")
                markdown_text = self.markdown.convert_url(data)
                logger.debug(f"Converted YouTube URL to markdown (length: {len(markdown_text)} chars)")
                
                return {
                    "data": [markdown_text],
                    "metadata": metadata or {}
                }
            else:
                error_msg = "Non-YouTube URLs are not supported at this time"
                logger.warning(f"{error_msg}: {data}")
                raise NotImplementedError(error_msg)
                
        except Exception as e:
            error_msg = f"Error processing URL: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise