import logging
import io
import os
import subprocess
from typing import Dict, Any, List, Optional
from PIL import Image
import pypdf
from pdf2image import convert_from_bytes
import base64
from magika import Magika
import sys
import asyncio
import csv
import json
import uuid
import os
from concurrent.futures import ProcessPoolExecutor

from app.processors.base_processor import BaseProcessor
from app.core.markitdown.markdown_handler import MarkDown
from app.core.model.model_handler import ModelHandler
from app.core.storage.s3_handler import S3Handler
from app.config import settings
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
        logger.info("FileProcessor initialized")

        self.unoserver_host = settings.UNOSERVER_HOST
        self.unoserver_port = settings.UNOSERVER_PORT

        total_cpus = os.cpu_count() or 1
        reserved = 2 if total_cpus > 2 + 1 else 1
        max_workers = max(1, total_cpus - reserved)
        self._executor = ProcessPoolExecutor(max_workers=max_workers)
        
        #TODO: Decide on what all file can be processed depending on Markitdown and PDF Converion libraries
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
    
    def detect_file_type(self, file_data: bytes) -> str:
        try:
            file_stream = io.BytesIO(file_data)
            kind = self.magika.identify_stream(file_stream)
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
    
    async def process_single_page(self, page_data: bytes, page_num: int) -> Dict[str, Any]:
        try:
            loop = asyncio.get_running_loop()
            extracted_text = self.markdown.convert_bytes(page_data)
            _, image_base64 = await loop.run_in_executor(
                self._executor, _rasterize_and_encode, page_data, page_num
            )
            text = await self.model_handler.generate_text_description(image_base64)
            
            text = text + "\n Extracted Text from this document: " + extracted_text

            return {
                "image": image_base64,
                "text": text or f"No text extracted from page {page_num + 1}"
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
    
            page_tasks:List[asyncio.Future] = []
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
    
    def convert_to_pdf(self, file_data: bytes, file_type: str) -> bytes:
        if file_type == 'pdf':
            return file_data
            
        try:
            input_bytes = io.BytesIO(file_data)
            input_bytes.seek(0)
            
            temp_input_path = "/tmp/input_document"
            temp_output_path = "/tmp/output_document.pdf"
            
            with open(temp_input_path, 'wb') as temp_file:
                temp_file.write(input_bytes.getvalue())
            
            logger.info(f"Converting file to PDF using unoserver on {self.unoserver_host}:{self.unoserver_port}")
            
            cmd = [
                "unoconvert",
                "--host-location", "remote",
                "--host", self.unoserver_host,
                "--port", self.unoserver_port,
                temp_input_path,
                temp_output_path
            ]
            
            process = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True,
                check=True
            )
            
            with open(temp_output_path, 'rb') as output_file:
                pdf_data = output_file.read()
            
            if os.path.exists(temp_input_path):
                os.remove(temp_input_path)
            if os.path.exists(temp_output_path):
                os.remove(temp_output_path)
            
            logger.info(f"File successfully converted to PDF")
            return pdf_data

        except subprocess.CalledProcessError as e:
            logger.error(f"Error in unoconvert process: {e.stdout} {e.stderr}")
            raise ValueError(f"Failed to convert file to PDF: {str(e)}")
        except Exception as e:
            logger.error(f"Error converting file to PDF: {e}")
            raise ValueError(f"Failed to convert file to PDF: {str(e)}")
    
    async def process_unstructured_document(self, file_data: bytes, file_type: str) -> Dict[str, Any]:
        file_data = self.convert_to_pdf(file_data, file_type)

        if self.detect_file_type(file_data) != 'pdf':
            error_msg = f"File conversion to PDF failed: {file_type}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        images_with_text = await self.convert_to_images(file_data)
        
        return {
            "data": images_with_text
        }
    
    def process_structured_document(self, file_data: bytes, file_type: str) -> Dict[str, Any]:
        try:
            markdown_text = self.markdown.convert_bytes(file_data)

            #TODO: Determine a strategy to proerly handle large files
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
                except Exception as e:
                    logger.debug(f"CSV header detection failed: {e}")
                if has_header:
                    logger.info("CSV header detected; applying to all batches")
                header = lines[0] if lines else ''
                data_lines = lines[1:] if len(lines) > 1 else []
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
    
    def process_direct_document(self, file_data: bytes, file_type: str) -> Dict[str, Any]:
        try:
            text = file_data.decode('utf-8', errors='replace')
            if file_type == 'markdown':
                return {
                    "data": [text]
                }
            
            markdown_text = self.markdown.convert_text(text)
            return {
                "data": [markdown_text]
            }
        except Exception as e:
            logger.error(f"Error processing direct document: {e}")
            raise

    async def process(self, data: bytes, metadata: Optional[Dict[str, Any]] = None, source: Optional[str] = None) -> Dict[str, Any]:
        logger.info("Processing file")
        
        if not data or not isinstance(data, bytes):
            error_msg = f"Invalid data type for file processing: {type(data)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        try:
            file_type = self.detect_file_type(data)
            logger.debug(f"Detected file type: {file_type}")
            
            category = self.categorize_file(file_type)
            logger.debug(f"File category: {category}")
            
            if category == 'unstructured':
                result = await self.process_unstructured_document(data, file_type)
                
            elif category == 'structured':
                result = self.process_structured_document(data, file_type)
            elif category == 'direct':
                result = self.process_direct_document(data, file_type)
            else:
                error_msg = f"Unsupported file type: {file_type}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            result["metadata"] = metadata or {}
            result['metadata']['file_type'] = file_type

            internal_object_id = str(uuid.uuid4())
            result['metadata']['internal_object_id'] = internal_object_id
            source_filename_from_meta = result['metadata'].get('filename', f'unknown_source_{internal_object_id}')
            
            base_filename_for_folder = os.path.splitext(source_filename_from_meta)[0]
            s3_friendly_folder_name = "".join(c if c.isalnum() or c in ['-', '_'] else '_' for c in base_filename_for_folder)
            if not s3_friendly_folder_name: 
                s3_friendly_folder_name = internal_object_id
            
            s3_base_path = f"{source}/{s3_friendly_folder_name}"

            page_image_s3_urls = []
            page_text_s3_urls = [] 

            if category == 'unstructured' and 'data' in result and isinstance(result['data'], list):
                for idx, page_content in enumerate(result['data']):
                    if 'image' in page_content and page_content['image']:
                        base64_image_string = page_content['image']
                        try:
                            image_bytes_data = base64.b64decode(base64_image_string)
                            image_s3_key = f"{s3_base_path}/page_{idx + 1}.jpg" 
                            upload_image_success = await self.s3_handler.upload_bytes(image_bytes_data, image_s3_key)
                            if upload_image_success:
                                s3_image_url = f"{self.s3_handler.endpoint_url}/{self.s3_handler.bucket_name}/{image_s3_key}"
                                page_image_s3_urls.append(s3_image_url)
                                logger.info(f"Successfully uploaded page {idx+1} image to S3: {s3_image_url}")
                            else:
                                logger.error(f"Failed to upload page {idx+1} image to S3 at {image_s3_key}.")
                        except Exception as img_e:
                            logger.error(f"Error decoding or uploading page {idx+1} image to S3 ({image_s3_key}): {img_e}")
                         
                    if 'text' in page_content and page_content['text']:
                        page_text_content = page_content['text']
                        text_s3_key = f"{s3_base_path}/page_{idx + 1}.txt"
                        try:
                            upload_text_success = await self.s3_handler.upload_string(page_text_content, text_s3_key)
                            if upload_text_success:
                                s3_text_page_url = f"{self.s3_handler.endpoint_url}/{self.s3_handler.bucket_name}/{text_s3_key}"
                                page_text_s3_urls.append(s3_text_page_url)
                                logger.info(f"Successfully uploaded page {idx+1} text to S3: {s3_text_page_url}")
                            else:
                                logger.error(f"Failed to upload page {idx+1} text to S3 at {text_s3_key}.")
                        except Exception as txt_e:
                            logger.error(f"Error uploading page {idx+1} text to S3 ({text_s3_key}): {txt_e}")
                
                result['metadata']['page_image_s3_urls'] = page_image_s3_urls
                result['metadata']['page_text_s3_urls'] = page_text_s3_urls
            
            return result
        except Exception as e:
            error_msg = f"Error processing file: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise
    
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

async def main():
    import os
    import json
    from pathlib import Path
    
    file_path = Path('pre-tests/Vivek Malipatel - Resu me.docx')
    if not file_path.exists():
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    
    processor = FileProcessor()
    await processor.s3_handler.initialize()


    with open(file_path, 'rb') as f:
        file_data = f.read()
    
    print(f"Processing file: {file_path.name}")
    try:
        result = await processor.process(file_data, metadata={"source_filename": file_path.name})

        print("\n--- Overall Processing Results ---")
        print(f"Processed {len(result['data'])} pages/images")

        if result['data'] and len(result['data']) > 0:
            sample_text = result['data'][0]['text']
            print(f"\nSample text from first page (truncated to 150 chars):")
            print(f"{sample_text[:150]}...")

            image_size = len(result['data'][0]['image'])
            print(f"\nBase64 image size for first page: {image_size} characters")

        output_dir = Path('output')
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / f"{file_path.stem}_processed.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
            
        print(f"\nFull results saved to: {output_file}")
            
        return result
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        raise

if __name__ == "__main__":
    result = asyncio.run(main())
    print("\nTest completed successfully!")