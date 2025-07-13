import io
import os
import subprocess
import logging
import asyncio
import csv
import base64
import uuid
from typing import Dict, Any, Optional, List, AsyncGenerator
from concurrent.futures import ProcessPoolExecutor

import pypdf
from magika import Magika
from pdf2image import convert_from_bytes

from core.processors.base_processor import BaseProcessor
from core.markitdown.markdown_handler import MarkDown
from core.model.model_handler import ModelHandler
from core.storage.s3_handler import S3Handler
from config import settings

logger = logging.getLogger(__name__)

def _rasterize_and_encode(page_bytes: bytes, page_num: int) -> tuple[int, str]:
    images = convert_from_bytes(page_bytes, dpi=300)
    img = images[0]
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return page_num, b64

class FileProcessor(BaseProcessor):
    
    def __init__(self):
        self.markdown = MarkDown()
        self.magika = Magika()
        self.model_handler = ModelHandler()
        self.s3_handler = S3Handler()
        
        self.unoserver_host = settings.UNOSERVER_HOST
        self.unoserver_port = settings.UNOSERVER_PORT

        total_cpus = os.cpu_count() or 1
        reserved = 2 if total_cpus > 2 + 1 else 1
        max_workers = max(1, total_cpus - reserved)
        self._executor = ProcessPoolExecutor(max_workers=max_workers)
        
        self.unstructured_docs = [
            'pdf', 
            'doc',
            'docx',
            'ppt',
            'txtascii',
            'txtutf8',
            'txtutf16',
            'pptx',
            'jpeg',
            'jpg',
            'png'
        ]
        
        self.structured_docs = [
            'csv',
            'xls',
            'xlsx',
            'txt'
        ]

        self.direct_processing_docs = [
            'markdown',
            'json',
            'python',
            'java',
            'go',
            'ruby',
            'php',
            'bash',
            'shell',
            'c',
            'javascript',
            'cpp',
            'html',
            'css',
            'xml',
            'yaml',
            'toml',
        ]
    
    async def detect_file_type(self, file_data: bytes) -> str:
        try:
            loop = asyncio.get_running_loop()
            file_stream = io.BytesIO(file_data)
            kind = await loop.run_in_executor(None, self.magika.identify_stream, file_stream)
            if kind is not None:
                return kind.output.label
            raise ValueError("Could not detect file type")
        except Exception as e:
            logger.error(f"Error detecting file type: {e}")
            return ''
    
    def categorize_file(self, file_type: str) -> str:
        if file_type in self.unstructured_docs:
            return 'unstructured'
        elif file_type in self.structured_docs:
            return 'structured'
        elif file_type in self.direct_processing_docs:
            return 'direct'
        else:
            raise ValueError(f"Unsupported File type: {file_type}")
    
    async def convert_to_pdf(self, file_data: bytes, file_type: str) -> bytes:
        if file_type == 'pdf':
            return file_data
            
        try:
            loop = asyncio.get_running_loop()
            
            def _convert_pdf():
                input_bytes = io.BytesIO(file_data)
                input_bytes.seek(0)
                
                temp_input_path = "/tmp/input_document"
                temp_output_path = "/tmp/output_document.pdf"
                
                with open(temp_input_path, 'wb') as temp_file:
                    temp_file.write(input_bytes.getvalue())
                
                cmd = [
                    "unoconvert",
                    "--host-location", "remote",
                    "--host", self.unoserver_host,
                    "--port", self.unoserver_port,
                    temp_input_path,
                    temp_output_path
                ]
                
                subprocess.run(cmd, capture_output=True, text=True, check=True)
                
                with open(temp_output_path, 'rb') as output_file:
                    pdf_data = output_file.read()
                
                if os.path.exists(temp_input_path):
                    os.remove(temp_input_path)
                if os.path.exists(temp_output_path):
                    os.remove(temp_output_path)
                
                return pdf_data
            
            return await loop.run_in_executor(None, _convert_pdf)
        except Exception as e:
            logger.error(f"Error converting file to PDF: {e}")
            raise ValueError(f"Failed to convert file to PDF: {str(e)}")

    async def process_single_page(self, page_data: bytes, page_num: int) -> Dict[str, Any]:
        try:
            loop = asyncio.get_running_loop()
            extracted_text_task = loop.run_in_executor(None, self.markdown.convert_bytes, page_data)
            image_task = loop.run_in_executor(self._executor, _rasterize_and_encode, page_data, page_num)
            
            extracted_text, (_, image_base64) = await asyncio.gather(extracted_text_task, image_task)
            text_description = await self.model_handler.generate_text_description(image_base64)
            
            full_text = text_description + "\n Extracted Text from this document: " + extracted_text

            return {
                "image": image_base64,
                "text": full_text or f"No text extracted from page {page_num + 1}"
            }
        except Exception as e:
            logger.error(f"Error processing page {page_num + 1}: {e}")
            return {
                "image": "",
                "text": f"Error processing page {page_num + 1}: {str(e)}"
            }

    async def convert_to_images(self, file_data: bytes) -> List[Dict[str, Any]]:
        try:
            pdf_stream = io.BytesIO(file_data)
            pdf_reader = pypdf.PdfReader(pdf_stream)
            total_pages = len(pdf_reader.pages)
            logger.info(f"Processing PDF with {total_pages} pages")
    
            page_tasks = []
            for page_num in range(total_pages):
                writer = pypdf.PdfWriter()
                writer.add_page(pdf_reader.pages[page_num])
                single_page_pdf = io.BytesIO()
                writer.write(single_page_pdf)
                single_page_pdf.seek(0)
                
                task = self.process_single_page(single_page_pdf.getvalue(), page_num)
                page_tasks.append(task)
            
            result = await asyncio.gather(*page_tasks)
            return result
                
        except Exception as e:
            logger.error(f"Error converting file to images: {e}")
            return [{
                "image": "",
                "text": f"Error converting file: {str(e)}"
            }]
    
    async def process_unstructured_document(self, file_data: bytes, file_type: str) -> Dict[str, Any]:
        file_data = await self.convert_to_pdf(file_data, file_type)

        detected_type = await self.detect_file_type(file_data)
        if detected_type != 'pdf':
            error_msg = f"File conversion to PDF failed: {file_type}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        images_with_text = await self.convert_to_images(file_data)
        
        return {
            "data": images_with_text
        }
    
    async def process_structured_document(self, file_data: bytes, file_type: str) -> Dict[str, Any]:
        try:
            loop = asyncio.get_running_loop()
            markdown_text = await loop.run_in_executor(None, self.markdown.convert_bytes, file_data)

            MAX_CHARS = 8000
            batches = []
            
            if len(markdown_text) <= MAX_CHARS:
                batches.append(markdown_text)
            else:
                lines = markdown_text.splitlines(True)
                has_header = False
                try:
                    sample = ''.join(lines[:min(len(lines), 10)])
                    has_header = csv.Sniffer().has_header(sample)
                except Exception:
                    pass
                    
                header = lines[0] if lines and has_header else ''
                data_lines = lines[1:] if len(lines) > 1 and has_header else lines
                current_batch = []
                current_length = len(header)
                
                for line in data_lines:
                    if current_length + len(line) > MAX_CHARS and current_batch:
                        batches.append(header + ''.join(current_batch))
                        current_batch = [line]
                        current_length = len(header) + len(line)
                    else:
                        current_batch.append(line)
                        current_length += len(line)
                        
                if current_batch:
                    batches.append(header + ''.join(current_batch))

            return {
                "data": batches
            }
        except Exception as e:
            logger.error(f"Error processing structured document: {e}")
            raise
    
    async def process_direct_document(self, file_data: bytes, file_type: str) -> Dict[str, Any]:
        try:
            loop = asyncio.get_running_loop()
            text = await loop.run_in_executor(None, lambda: file_data.decode('utf-8', errors='replace'))
            
            if file_type == 'markdown':
                return {
                    "data": [text]
                }
            
            markdown_text = await loop.run_in_executor(None, self.markdown.convert_text, text)
            return {
                "data": [markdown_text]
            }
        except Exception as e:
            logger.error(f"Error processing direct document: {e}")
            raise

    async def upload_page_data(self, page_content: Dict[str, Any], idx: int, s3_base_path: str) -> Dict[str, str]:
        upload_tasks = []
        urls = {"image_url": "", "text_url": ""}
        
        if 'image' in page_content and page_content['image']:
            image_bytes_data = base64.b64decode(page_content['image'])
            image_s3_key = f"{s3_base_path}/page_{idx + 1}.jpg"
            upload_tasks.append(self.s3_handler.upload_bytes(image_bytes_data, image_s3_key))
            urls["image_url"] = f"{self.s3_handler.endpoint_url}/{self.s3_handler.bucket_name}/{image_s3_key}"
        
        if 'text' in page_content and page_content['text']:
            text_s3_key = f"{s3_base_path}/page_{idx + 1}.txt"
            upload_tasks.append(self.s3_handler.upload_string(page_content['text'], text_s3_key))
            urls["text_url"] = f"{self.s3_handler.endpoint_url}/{self.s3_handler.bucket_name}/{text_s3_key}"
        
        if upload_tasks:
            await asyncio.gather(*upload_tasks)
        
        return urls

    async def process(self, data: bytes, metadata: Optional[Dict[str, Any]] = None, source: Optional[str] = None) -> AsyncGenerator[Dict[str, Any], None]:
        if not data or not isinstance(data, bytes):
            error_msg = f"Invalid data type for file processing: {type(data)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        try:
            file_type = await self.detect_file_type(data)
            category = self.categorize_file(file_type)
            
            if metadata is None:
                metadata = {}
            
            metadata["file_type"] = file_type
            
            if source is None:
                source = "unknown_source"
            
            if category == 'unstructured':
                result = await self.process_unstructured_document(data, file_type)
            elif category == 'structured':
                result = await self.process_structured_document(data, file_type)
            elif category == 'direct':
                result = await self.process_direct_document(data, file_type)
            else:
                error_msg = f"Unsupported file type: {file_type}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            result["metadata"] = metadata.copy()
            result['metadata']['file_type'] = file_type

            internal_object_id = str(uuid.uuid4())
            result['metadata']['internal_object_id'] = internal_object_id
            source_filename_from_meta = result['metadata'].get('filename', f'unknown_source_{internal_object_id}')
            
            base_filename_for_folder = os.path.splitext(source_filename_from_meta)[0]
            s3_friendly_folder_name = "".join(c if c.isalnum() or c in ['-', '_'] else '_' for c in base_filename_for_folder)
            if not s3_friendly_folder_name: 
                s3_friendly_folder_name = internal_object_id
            
            s3_base_path = f"{source}/{s3_friendly_folder_name}"

            if category == 'unstructured' and 'data' in result and isinstance(result['data'], list):
                upload_tasks = []
                for idx, page_content in enumerate(result['data']):
                    upload_tasks.append(self.upload_page_data(page_content, idx, s3_base_path))
                
                upload_results = await asyncio.gather(*upload_tasks)
                
                page_image_s3_urls = []
                page_text_s3_urls = []
                
                for upload_result in upload_results:
                    if upload_result["image_url"]:
                        page_image_s3_urls.append(upload_result["image_url"])
                    if upload_result["text_url"]:
                        page_text_s3_urls.append(upload_result["text_url"])
                
                result['metadata']['page_image_s3_urls'] = page_image_s3_urls
                result['metadata']['page_text_s3_urls'] = page_text_s3_urls
            
            # Yield individual items
            if 'data' in result:
                for idx, item_data in enumerate(result['data']):
                    if isinstance(item_data, str):
                        item_text = item_data
                        item = {
                            "text": item_text,
                            "metadata": result['metadata'].copy()
                        }
                    elif isinstance(item_data, dict):
                        item = item_data.copy()
                        if "image" in item:
                            item["image_b64"] = item.pop("image")
                        if "metadata" not in item:
                            item["metadata"] = result['metadata'].copy()
                    else:
                        continue
                    

                    item['page'] = idx + 1
                    item['metadata']['page_number'] = idx + 1
                    item['metadata']['total_pages'] = len(result['data'])

                    yield item
            
        except Exception as e:
            error_msg = f"Error processing file: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise