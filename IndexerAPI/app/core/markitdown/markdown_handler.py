import io
import os
from typing import Optional, Union, BinaryIO
from pathlib import Path
import requests
from openai import OpenAI
from markitdown import MarkItDown
from markitdown._base_converter import DocumentConverterResult
import base64


class MarkDown:
    def __init__(self, enable_plugins: bool = False, api_key: str = None, api_base: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.api_base = api_base or os.getenv("OPENAI_API_BASE")
        self.cleint = OpenAI(api_key=self.api_key, base_url=self.api_base)
        self.markitdown = MarkItDown(enable_plugins=enable_plugins, llm_client=self.cleint, llm_model="gemma3:12b-it-q8_0")

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
        stream = io.BytesIO(text.encode('utf-8'))
        result = self.markitdown.convert_stream(stream, file_extension='.txt')
        return result.text_content

    def convert_html(self, html_content: str) -> str:
        stream = io.BytesIO(html_content.encode('utf-8'))
        result = self.markitdown.convert_stream(stream, file_extension='.html')
        return result.text_content

    def get_raw_result(self, file_path: Union[str, Path, bytes, BinaryIO, str]) -> DocumentConverterResult:
        if isinstance(file_path, (str, Path)) and os.path.exists(str(file_path)):
            return self.markitdown.convert_local(file_path)
        elif isinstance(file_path, bytes):
            stream = io.BytesIO(file_path)
            return self.markitdown.convert_stream(stream)
        elif hasattr(file_path, 'read') and callable(file_path.read):
            return self.markitdown.convert_stream(file_path)
        elif isinstance(file_path, str) and (file_path.startswith('http://') or file_path.startswith('https://')):
            return self.markitdown.convert_url(file_path)
        else:
            if isinstance(file_path, str):
                stream = io.BytesIO(file_path.encode('utf-8'))
                return self.markitdown.convert_stream(stream, file_extension='.txt')
            raise ValueError("Unsupported input type")


def main():
    try:
        # # Example: Convert an image file using base64 encoding
        # image_path = "pre-tests/test.png" # Change this to your image file path

        # if not os.path.exists(image_path):
        #     print(f"Error: File '{image_path}' not found. Please provide a valid path to an image file.")
        #     return

        # print(f"Converting '{image_path}' to markdown...")

        # # Read the image file and convert to base64
        # with open(image_path, "rb") as image_file:
        #     image_bytes = image_file.read()
        #     # Prepend the data URI scheme if needed by the specific implementation
        #     # base64_encoded = f"data:image/png;base64,{base64.b64encode(image_bytes).decode('utf-8')}"
        #     base64_encoded = base64.b64encode(image_bytes).decode('utf-8')


        # md = MarkDown()

        # # Determine the file extension from the path
        # _, file_extension = os.path.splitext(image_path)

        # # Use convert_binary_stream with the base64 string and correct extension
        # markdown_text = md.convert_binary_stream(base64_encoded, file_extension=file_extension)

        # print("\nConverted Markdown:")
        # print("-" * 50)
        # print(markdown_text)
        # print("-" * 50)

        # Example: Convert a PDF file to Markdown and save it
        pdf_path = "pre-tests/1744918052438.pdf" # Change this to your PDF file path

        if not os.path.exists(pdf_path):
            print(f"Error: File '{pdf_path}' not found. Please provide a valid path to a PDF file.")
            return

        print(f"Converting '{pdf_path}' to markdown...")

        md = MarkDown()

        # Convert the PDF file to markdown text
        markdown_text = md.convert_file(pdf_path)

        # Determine the output markdown file path
        base_name, _ = os.path.splitext(pdf_path)
        markdown_output_path = f"{base_name}.md"

        # Save the markdown text to the output file
        with open(markdown_output_path, "w", encoding="utf-8") as md_file:
            md_file.write(markdown_text)

        print(f"\nConverted Markdown saved to: '{markdown_output_path}'")
        print("-" * 50)
        # Optionally print the markdown content as well
        # print(markdown_text)
        # print("-" * 50)
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()