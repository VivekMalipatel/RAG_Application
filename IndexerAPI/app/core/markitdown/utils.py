"""
Utility functions for working with MarkItDown conversions.
"""

import os
import logging
import tempfile
from typing import Dict, Any, Optional
from fastapi import UploadFile

from app.core.markitdown.converter import MarkdownConverter

logger = logging.getLogger(__name__)

async def convert_upload_file_to_markdown(
    upload_file: UploadFile, 
    enable_plugins: bool = False
) -> Dict[str, Any]:
    """
    Convert an uploaded file to markdown.
    
    Args:
        upload_file: FastAPI UploadFile object
        enable_plugins: Whether to enable MarkItDown plugins
        
    Returns:
        Dict containing:
            text_content: Markdown content
            metadata: Metadata from the conversion
    """
    try:
        logger.info(f"Converting uploaded file to markdown: {upload_file.filename}")
        converter = MarkdownConverter(enable_plugins=enable_plugins)
        
        # Save the file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(upload_file.filename)[1]) as temp_file:
            # Copy content from upload file to temp file
            content = await upload_file.read()
            temp_file.write(content)
            temp_file.flush()
            
            # Convert the temp file
            result = converter.convert_file(temp_file.name)
            
            # Delete the temp file
            os.unlink(temp_file.name)
        
        # Reset the file pointer so it can be read again if needed
        await upload_file.seek(0)
        return result
    
    except Exception as e:
        logger.error(f"Error converting upload file to markdown: {str(e)}")
        return {"text_content": "", "metadata": {"error": str(e)}}

async def convert_file_bytes_to_markdown(
    file_content: bytes, 
    file_extension: str,
    enable_plugins: bool = False
) -> Dict[str, Any]:
    """
    Convert file bytes to markdown.
    
    Args:
        file_content: Binary file content
        file_extension: The file extension including the dot (e.g., '.pdf')
        enable_plugins: Whether to enable MarkItDown plugins
        
    Returns:
        Dict containing:
            text_content: Markdown content
            metadata: Metadata from the conversion
    """
    try:
        logger.info(f"Converting file bytes to markdown with extension: {file_extension}")
        converter = MarkdownConverter(enable_plugins=enable_plugins)
        
        # Convert file bytes using convert_stream
        import io
        file_stream = io.BytesIO(file_content)
        result = converter.convert_stream(file_stream, file_extension)
        
        return result
    
    except Exception as e:
        logger.error(f"Error converting file bytes to markdown: {str(e)}")
        return {"text_content": "", "metadata": {"error": str(e)}}

def extract_text_from_markdown(markdown_text: str) -> str:
    """
    Extract plain text from markdown, removing markdown formatting.
    This is a simple implementation - for more complex cases, consider using a markdown parser.
    
    Args:
        markdown_text: Markdown formatted text
        
    Returns:
        Plain text with most markdown formatting removed
    """
    if not markdown_text:
        return ""
        
    # Remove headers (# Header)
    lines = markdown_text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # Skip horizontal rules
        if line.strip() in ['---', '***', '___']:
            continue
            
        # Remove headers
        line = line.lstrip('#').lstrip()
        
        # Remove bold and italic
        line = line.replace('**', '').replace('__', '').replace('*', '').replace('_', '')
        
        # Remove inline code
        while '`' in line:
            line = line.replace('`', '', 1)
            if '`' in line:
                line = line.replace('`', '', 1)
        
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)