import io
import os
from typing import Optional, Union, BinaryIO
from pathlib import Path
import base64
from markitdown import MarkItDown
from markitdown._base_converter import DocumentConverterResult

class MarkDown:
    def __init__(self, enable_plugins: bool = False):
        self.markitdown = MarkItDown(enable_plugins=enable_plugins, llm_client=None)

    def convert_file(self, file_path: Union[str, Path]) -> str:
        result = self.markitdown.convert_local(file_path)
        return result.text_content

    def convert_bytes(self, file_bytes: bytes, file_extension: Optional[str] = None) -> str:
        stream = io.BytesIO(file_bytes)
        result = self.markitdown.convert_stream(stream, file_extension=file_extension)
        return result.text_content

    def convert_binary_stream(self, stream: Union[BinaryIO, str], file_extension: Optional[str] = None) -> str:
        if isinstance(stream, str):
            if ";" in stream and "," in stream:
                stream = stream.split(",", 1)[1]
            binary_data = base64.b64decode(stream)
            stream = io.BytesIO(binary_data)
        result = self.markitdown.convert_stream(stream, file_extension=file_extension)
        return result.text_content

    def convert_url(self, url: str) -> str:
        result = self.markitdown.convert_url(url)
        return result.text_content

    def convert_text(self, text: str) -> str:
        stream = io.BytesIO(text.encode("utf-8"))
        result = self.markitdown.convert_stream(stream, file_extension=".txt")
        return result.text_content

    def convert_html(self, html_content: str) -> str:
        stream = io.BytesIO(html_content.encode("utf-8"))
        result = self.markitdown.convert_stream(stream, file_extension=".html")
        return result.text_content

    def get_raw_result(self, file_path: Union[str, Path, bytes, BinaryIO, str]) -> DocumentConverterResult:
        if isinstance(file_path, (str, Path)) and os.path.exists(str(file_path)):
            return self.markitdown.convert_local(file_path)
        if isinstance(file_path, bytes):
            stream = io.BytesIO(file_path)
            return self.markitdown.convert_stream(stream)
        if hasattr(file_path, "read") and callable(file_path.read):
            return self.markitdown.convert_stream(file_path)
        if isinstance(file_path, str) and (file_path.startswith("http://") or file_path.startswith("https://")):
            return self.markitdown.convert_url(file_path)
        if isinstance(file_path, str):
            stream = io.BytesIO(file_path.encode("utf-8"))
            return self.markitdown.convert_stream(stream, file_extension=".txt")
        raise ValueError("Unsupported input type")
