import logging
import os
import io
from pathlib import Path
from typing import Dict, Any, Optional, Union, BinaryIO
from markitdown import MarkItDown


logger = logging.getLogger(__name__)

class MarkdownConverter:
    """
    Handles conversion of various file formats to Markdown using Microsoft's MarkItDown library.
    Supports PDF, PowerPoint, Word, Excel, Images, Audio, HTML, and more.
    """

    def __init__(self, enable_plugins: bool = False, use_azure: bool = False, azure_endpoint: Optional[str] = None):
        """
        Initialize the MarkdownConverter.
        
        Args:
            enable_plugins: Whether to enable MarkItDown plugins
            use_azure: Whether to use Azure Document Intelligence
            azure_endpoint: Azure Document Intelligence endpoint if use_azure is True
        """
        self.enable_plugins = enable_plugins
        self.use_azure = use_azure
        self.azure_endpoint = azure_endpoint
            
        # Initialize MarkItDown instance
        self._markitdown = MarkItDown(enable_plugins=enable_plugins)
        logger.info("MarkdownConverter initialized with MarkItDown")
        
    def convert_file(self, file_path: str) -> Dict[str, Any]:
        """
        Convert a file to Markdown using MarkItDown.
        
        Args:
            file_path: Path to the file to convert
            
        Returns:
            Dict containing:
                text_content: The markdown content
                metadata: Additional metadata from the conversion
        """
            
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return {"text_content": "", "metadata": {"error": f"File not found: {file_path}"}}
            
        try:
            logger.info(f"Converting file to markdown: {file_path}")
            
            # Use Azure Document Intelligence if specified
            if self.use_azure and self.azure_endpoint:
                result = self._markitdown.convert(
                    file_path, 
                    use_document_intelligence=True,
                    document_intelligence_endpoint=self.azure_endpoint
                )
            else:
                result = self._markitdown.convert(file_path)
                
            logger.info(f"Successfully converted file to markdown: {file_path}")
            
            # Return the markdown content and metadata
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
        """
        Convert a file stream to Markdown.
        
        Args:
            file_stream: Binary file-like object
            file_extension: File extension (e.g., '.pdf', '.docx')
            
        Returns:
            Dict containing:
                text_content: The markdown content
                metadata: Additional metadata from the conversion
        """
            
        try:
            logger.info(f"Converting stream to markdown with extension: {file_extension}")
            
            # MarkItDown convert_stream requires binary data
            result = self._markitdown.convert_stream(file_stream, file_extension=file_extension)
            
            logger.info("Successfully converted stream to markdown")
            
            # Return the markdown content and metadata
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
        """
        Convert content from a URL to Markdown.
        MarkItDown has special handling for YouTube URLs.
        
        Args:
            url: The URL to convert
            
        Returns:
            Dict containing:
                text_content: The markdown content
                metadata: Additional metadata from the conversion
        """
            
        try:
            logger.info(f"Converting URL to markdown: {url}")
            
            # Handle URL conversion
            result = self._markitdown.convert(url)
            
            logger.info(f"Successfully converted URL to markdown: {url}")
            
            # Return the markdown content and metadata
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