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
    try:
        logger.info(f"Converting uploaded file to markdown: {upload_file.filename}")
        converter = MarkdownConverter(enable_plugins=enable_plugins)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(upload_file.filename)[1]) as temp_file:
            content = await upload_file.read()
            temp_file.write(content)
            temp_file.flush()
            
            result = converter.convert_file(temp_file.name)
            
            os.unlink(temp_file.name)
        
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
    try:
        logger.info(f"Converting file bytes to markdown with extension: {file_extension}")
        converter = MarkdownConverter(enable_plugins=enable_plugins)
        
        import io
        file_stream = io.BytesIO(file_content)
        result = converter.convert_stream(file_stream, file_extension)
        
        return result
    
    except Exception as e:
        logger.error(f"Error converting file bytes to markdown: {str(e)}")
        return {"text_content": "", "metadata": {"error": str(e)}}

def extract_text_from_markdown(markdown_text: str) -> str:
    if not markdown_text:
        return ""
        
    lines = markdown_text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        if line.strip() in ['---', '***', '___']:
            continue
            
        line = line.lstrip('#').lstrip()
        
        line = line.replace('**', '').replace('__', '').replace('*', '').replace('_', '')
        
        while '`' in line:
            line = line.replace('`', '', 1)
            if '`' in line:
                line = line.replace('`', '', 1)
        
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)