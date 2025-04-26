import logging
import os
import io
from pathlib import Path
from typing import Dict, Any, Optional, Union, BinaryIO
from markitdown import MarkItDown


logger = logging.getLogger(__name__)

class MarkdownConverter:

    def __init__(self, enable_plugins: bool = False, use_azure: bool = False, azure_endpoint: Optional[str] = None):
        self.enable_plugins = enable_plugins
        self.use_azure = use_azure
        self.azure_endpoint = azure_endpoint
            
        self._markitdown = MarkItDown(enable_plugins=enable_plugins)
        logger.info("MarkdownConverter initialized with MarkItDown")
        
    def convert_file(self, file_path: str) -> Dict[str, Any]:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return {"text_content": "", "metadata": {"error": f"File not found: {file_path}"}}
            
        try:
            logger.info(f"Converting file to markdown: {file_path}")
            
            if self.use_azure and self.azure_endpoint:
                result = self._markitdown.convert(
                    file_path, 
                    use_document_intelligence=True,
                    document_intelligence_endpoint=self.azure_endpoint
                )
            else:
                result = self._markitdown.convert(file_path)
                
            logger.info(f"Successfully converted file to markdown: {file_path}")
            
            return {
                "text_content": result.text_content,
                "metadata": {
                    "file_type": result.file_type,
                    "page_count": getattr(result, "page_count", None),
                    "title": getattr(result, "title", None),
                    "author": getattr(result, "author", None),
                    "creation_date": getattr(result, "creation_date", None),
                }
            }
        except Exception as e:
            logger.error(f"Error converting file to markdown: {str(e)}")
            return {"text_content": "", "metadata": {"error": str(e)}}
            
    def convert_stream(self, file_stream: BinaryIO, file_extension: str) -> Dict[str, Any]:
        try:
            logger.info(f"Converting stream to markdown with extension: {file_extension}")
            
            result = self._markitdown.convert_stream(file_stream, file_extension=file_extension)
            
            logger.info("Successfully converted stream to markdown")
            
            return {
                "text_content": result.text_content,
                "metadata": {
                    "file_type": result.file_type,
                    "page_count": getattr(result, "page_count", None),
                    "title": getattr(result, "title", None),
                    "author": getattr(result, "author", None),
                    "creation_date": getattr(result, "creation_date", None),
                }
            }
        except Exception as e:
            logger.error(f"Error converting stream to markdown: {str(e)}")
            return {"text_content": "", "metadata": {"error": str(e)}}
    
    def convert_url(self, url: str) -> Dict[str, Any]:
        try:
            logger.info(f"Converting URL to markdown: {url}")
            
            result = self._markitdown.convert(url)
            
            logger.info(f"Successfully converted URL to markdown: {url}")
            
            return {
                "text_content": result.text_content,
                "metadata": {
                    "url": url,
                    "is_youtube": "youtube.com" in url or "youtu.be" in url,
                    "title": getattr(result, "title", None),
                }
            }
        except Exception as e:
            logger.error(f"Error converting URL to markdown: {str(e)}")
            return {"text_content": "", "metadata": {"error": str(e)}}