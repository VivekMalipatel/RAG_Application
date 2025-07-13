import logging
import re
from typing import Dict, Any, List, Optional, AsyncGenerator
import urllib.parse
import hashlib

from core.processors.base_processor import BaseProcessor
from core.markitdown.markdown_handler import MarkDown

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
    
    async def process(self, data: str, metadata: Optional[Dict[str, Any]] = None, source: str = None) -> AsyncGenerator[Dict[str, Any], None]:
        if not data or not isinstance(data, str):
            error_msg = f"Invalid data type for URL processing: {type(data)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        try:
            is_youtube = self.is_youtube_url(data)
            
            if is_youtube:
                text_content = await self._process_youtube_url(data)
            else:
                text_content = self.markdown.convert_url(data)
            
            chunks = self._split_text(text_content)
            
            for i, chunk in enumerate(chunks):
                item = {
                    "text": chunk,
                    "metadata": metadata.copy() if metadata else {},
                    "batch": i
                }
                yield item
            
        except Exception as e:
            logger.error(f"Error processing URL: {str(e)}")
            raise
            
    async def _process_youtube_url(self, url: str) -> str:
        try:
            return self.markdown.convert_url(url)
        except Exception as e:
            logger.error(f"Error processing YouTube URL: {e}")
            raise
    
    def _split_text(self, text: str, max_chunk_size: int = 4000) -> List[str]:
        if len(text) <= max_chunk_size:
            return [text]
        
        chunks = []
        current_position = 0
        text_length = len(text)
        
        while current_position < text_length:
            end_position = min(current_position + max_chunk_size, text_length)

            if end_position < text_length:
                paragraph_break = text.rfind('\n\n', current_position, end_position)
                if paragraph_break != -1 and paragraph_break > current_position + max_chunk_size//2:
                    end_position = paragraph_break + 2
                else:
                    newline = text.rfind('\n', current_position, end_position)
                    if newline != -1 and newline > current_position + max_chunk_size//2:
                        end_position = newline + 1
                    else:
                        for sep in ['. ', '! ', '? ']:
                            sentence_break = text.rfind(sep, current_position, end_position)
                            if sentence_break != -1 and sentence_break > current_position + max_chunk_size//2:
                                end_position = sentence_break + 2
                                break
            
            chunks.append(text[current_position:end_position].strip())
            current_position = end_position
            
        return chunks