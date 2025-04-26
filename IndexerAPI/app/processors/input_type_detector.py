import logging
import mimetypes
import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class InputTypeDetector:
    
    def __init__(self):
        mimetypes.init()
        
        mimetypes.add_type("text/markdown", ".md")
        mimetypes.add_type("text/markdown", ".markdown")
        
        logger.info("Input Type Detector initialized")
    
    def detect_file_type(self, file_path: str) -> str:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_ext = os.path.splitext(file_path)[1].lower()
        mime_type, _ = mimetypes.guess_type(file_path)
        
        logger.info(f"Detecting file type for {file_path}: ext={file_ext}, mime={mime_type}")
        
        if file_ext in ['.pdf'] or (mime_type and mime_type == 'application/pdf'):
            return "pdf"
        
        elif file_ext in ['.doc', '.docx', '.rtf', '.odt'] or (mime_type and 'document' in mime_type):
            return "document"
        
        elif file_ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'] or (mime_type and mime_type.startswith('image/')):
            return "image"
        
        elif file_ext in ['.txt', '.text'] or (mime_type and mime_type == 'text/plain'):
            return "text"
        
        elif file_ext in ['.md', '.markdown'] or (mime_type and mime_type == 'text/markdown'):
            return "markdown"
        
        elif file_ext in ['.html', '.htm'] or (mime_type and mime_type == 'text/html'):
            return "html"
        
        elif file_ext in ['.json'] or (mime_type and mime_type == 'application/json'):
            return "json"
        
        elif file_ext in ['.xml'] or (mime_type and mime_type == 'application/xml'):
            return "xml"
        
        elif file_ext in ['.csv'] or (mime_type and mime_type == 'text/csv'):
            return "csv"
        
        elif file_ext in ['.xls', '.xlsx', '.ods'] or (mime_type and 'spreadsheet' in mime_type):
            return "spreadsheet"
        
        elif file_ext in ['.mp3', '.wav', '.ogg', '.flac', '.m4a'] or (mime_type and mime_type.startswith('audio/')):
            return "audio"
        
        elif file_ext in ['.mp4', '.avi', '.mov', '.wmv', '.mkv', '.webm'] or (mime_type and mime_type.startswith('video/')):
            return "video"
        
        return self._detect_from_content(file_path)
    
    def _detect_from_content(self, file_path: str) -> str:
        try:
            with open(file_path, 'rb') as f:
                data = f.read(1024)
                
            if data.startswith(b'%PDF'):
                return "pdf"
            
            if data.startswith((b'\xff\xd8\xff', b'\x89PNG', b'GIF8', b'BM')):
                return "image"
            
            try:
                data.decode('utf-8')
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = [l.strip() for l in f.readlines(20)]
                    
                if any(line.startswith(('#', '##', '###', '####')) for line in lines):
                    return "markdown"
                
                if any(line.startswith('<') and line.endswith('>') for line in lines):
                    if any('<html' in line.lower() for line in lines):
                        return "html"
                    return "xml"
                    
                if any(line.startswith('{') or line.strip() == '[' for line in lines):
                    return "json"
                
                if ',' in lines[0] and len(lines[0].split(',')) > 1:
                    return "csv"
                
                return "text"
                
            except UnicodeDecodeError:
                return "binary"
                
        except Exception as e:
            logger.error(f"Error analyzing file content: {str(e)}")
            return "unknown"
    
    def is_url(self, text: str) -> bool:
        try:
            result = urlparse(text)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    def is_youtube_url(self, url: str) -> bool:
        if not self.is_url(url):
            return False
        
        youtube_patterns = [
            r'^(https?://)?(www\.)?(youtube\.com|youtu\.be|youtube-nocookie\.com)',
            r'youtube\.com/watch\?v=',
            r'youtu\.be/',
        ]
        
        for pattern in youtube_patterns:
            if re.search(pattern, url):
                return True
                
        return False
    
    def detect_input_type(self, input_data: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        if isinstance(input_data, str):
            input_str = input_data
            
            if self.is_url(input_str):
                if self.is_youtube_url(input_str):
                    return {
                        "type": "url",
                        "subtype": "youtube",
                        "url": input_str
                    }
                else:
                    return {
                        "type": "url",
                        "subtype": "web",
                        "url": input_str
                    }
                    
            if os.path.exists(input_str):
                file_type = self.detect_file_type(input_str)
                return {
                    "type": "file",
                    "subtype": file_type,
                    "path": input_str
                }
                
            return {
                "type": "text",
                "subtype": "raw",
                "content": input_str
            }
            
        elif isinstance(input_data, dict):
            if "url" in input_data:
                url = input_data["url"]
                if self.is_youtube_url(url):
                    return {
                        "type": "url",
                        "subtype": "youtube",
                        "url": url,
                        "metadata": input_data.get("metadata", {})
                    }
                else:
                    return {
                        "type": "url",
                        "subtype": "web",
                        "url": url,
                        "metadata": input_data.get("metadata", {})
                    }
                    
            if "text" in input_data or "content" in input_data:
                content = input_data.get("text", input_data.get("content", ""))
                return {
                    "type": "text",
                    "subtype": "raw",
                    "content": content,
                    "metadata": input_data.get("metadata", {})
                }
                
            if "file" in input_data or "path" in input_data:
                path = input_data.get("file", input_data.get("path", ""))
                if os.path.exists(path):
                    file_type = self.detect_file_type(path)
                    return {
                        "type": "file",
                        "subtype": file_type,
                        "path": path,
                        "metadata": input_data.get("metadata", {})
                    }
            
            return {
                "type": "unknown",
                "data": input_data
            }
            
        return {
            "type": "unknown",
            "error": "Unsupported input type"
        }