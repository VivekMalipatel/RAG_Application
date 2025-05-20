import io
import os
import subprocess
import logging
import asyncio
import csv
import threading
import base64
from typing import Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor

import pypdf
from magika import Magika
from pdf2image import convert_from_bytes

from app.processors.base_processor import BaseProcessor
from app.core.markitdown.markdown_handler import MarkDown
from app.core.model.model_handler import ModelHandler
from app.core.storage.s3_handler import S3Handler
from app.utils.thread_pool import get_cpu_thread_pool
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
        
        self.unoserver_host = settings.UNOSERVER_HOST
        self.unoserver_port = settings.UNOSERVER_PORT

        total_cpus = os.cpu_count() or 1
        reserved = 2 if total_cpus > 2 + 1 else 1
        max_workers = max(1, total_cpus - reserved)
        self._executor = ProcessPoolExecutor(max_workers=max_workers)
        self._thread_pool = get_cpu_thread_pool()
        self._prefetch_queue = asyncio.Queue(maxsize=20)
        
        self.default_batch_sizes = {"unstructured": 4, "structured": 3, "direct": 1}
        
        self.complexity_thresholds = {
            "page_count": {"low": 10, "medium": 50, "high": 100},
            "image_density": {"low": 0.2, "medium": 0.5, "high": 0.8},
            "text_density": {"low": 1000, "medium": 5000, "high": 10000}
        }
        
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
        
        self._document_metadata = {}
        self._prefetch_lock = threading.Lock()
    
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
        except Exception as e:
            logger.error(f"Error converting file to PDF: {e}")
            raise ValueError(f"Failed to convert file to PDF: {str(e)}")
    
    async def process_unstructured_document(self, file_data: bytes, file_type: str, source: str, metadata: Dict[str, Any]):
        file_data = self.convert_to_pdf(file_data, file_type)

        if self.detect_file_type(file_data) != 'pdf':
            error_msg = f"File conversion to PDF failed: {file_type}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        try:
            pdf_stream = io.BytesIO(file_data)
            pdf_reader = pypdf.PdfReader(pdf_stream)
            total_pages = len(pdf_reader.pages)
            
            complexity_metrics = self._analyze_document_complexity(pdf_reader)
            adaptive_batch_size = self._calculate_adaptive_batch_size('unstructured', complexity_metrics)
            
            base_filename = metadata.get('filename', f'unknown_source_{metadata.get("internal_object_id", "")}')
            s3_friendly_name = "".join(c if c.isalnum() or c in ['-', '_'] else '_' for c in base_filename)
            s3_base_path = f"{source}/{s3_friendly_name}"
            
            while not self._prefetch_queue.empty():
                try:
                    await self._prefetch_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                    
            prefetch_batch_size = adaptive_batch_size * 2
            asyncio.create_task(self._prefetch_pages(pdf_reader, 0, prefetch_batch_size, total_pages))
            
            for start_idx in range(0, total_pages, adaptive_batch_size):
                end_idx = min(start_idx + adaptive_batch_size, total_pages)
                batch_tasks = []
                
                next_batch_start = end_idx
                next_batch_end = min(next_batch_start + prefetch_batch_size, total_pages)
                if next_batch_start < total_pages:
                    asyncio.create_task(self._prefetch_pages(
                        pdf_reader, next_batch_start, next_batch_end, total_pages))
                
                for page_num in range(start_idx, end_idx):
                    prefetched_page = await self._get_prefetched_page(page_num)
                    
                    if prefetched_page:
                        page_num, page_data = prefetched_page
                    else:
                        writer = pypdf.PdfWriter()
                        writer.add_page(pdf_reader.pages[page_num])
                        single_page_pdf = io.BytesIO()
                        writer.write(single_page_pdf)
                        single_page_pdf.seek(0)
                        page_data = single_page_pdf.getvalue()
                    
                    task = self._process_single_page(
                        page_data=page_data,
                        page_num=page_num,
                        total_pages=total_pages,
                        s3_base_path=s3_base_path,
                        metadata=metadata
                    )
                    batch_tasks.append(task)
                
                results = await asyncio.gather(*batch_tasks)
                
                for result in results:
                    if result:
                        yield result
        except Exception as e:
            logger.error(f"Error processing unstructured document: {e}")
            raise
    
    async def _process_single_page(self, page_data: bytes, page_num: int, total_pages: int, 
                                  s3_base_path: str, metadata: Dict[str, Any]):
        try:
            loop = asyncio.get_running_loop()
            
            extraction_task = loop.run_in_executor(None, self.markdown.convert_bytes, page_data)
            
            cpu_pool = get_cpu_thread_pool()
            rasterize_future = cpu_pool.submit(_rasterize_and_encode, page_data, page_num)
            rasterize_task = loop.run_in_executor(None, lambda: rasterize_future.result())
            
            extracted_text, (_, image_base64) = await asyncio.gather(extraction_task, rasterize_task)
            text_description = await self.model_handler.generate_text_description(image_base64)
            full_text = text_description + "\n Extracted Text from this document: " + extracted_text
            
            upload_tasks = []
            
            try:
                image_bytes = base64.b64decode(image_base64)
                image_s3_key = f"{s3_base_path}/page_{page_num + 1}.jpg"
                upload_tasks.append(self.s3_handler.upload_bytes(image_bytes, image_s3_key))
            except Exception as img_e:
                logger.error(f"Error preparing image for page {page_num + 1}: {img_e}")
            
            try:
                text_s3_key = f"{s3_base_path}/page_{page_num + 1}.txt"
                upload_tasks.append(self.s3_handler.upload_string(full_text, text_s3_key))
            except Exception as txt_e:
                logger.error(f"Error preparing text for page {page_num + 1}: {txt_e}")
            
            if upload_tasks:
                await asyncio.gather(*upload_tasks)
            
            page_metadata = metadata.copy()
            page_metadata.update({
                "page_number": page_num + 1,
                "total_pages": total_pages,
                "document_id": metadata.get("internal_object_id", "")
            })
            
            return {
                "page": page_num + 1,
                "text": full_text,
                "image_b64": image_base64,
                "metadata": page_metadata
            }
        except Exception as e:
            logger.error(f"Error processing page {page_num + 1}: {e}")
            return None
    
    async def process_structured_document(self, file_data: bytes, file_type: str, source: str, metadata: Dict[str, Any]):
        try:
            markdown_text = self.markdown.convert_bytes(file_data)

            #TODO: Determine a strategy to properly handle large files
            MAX_CHARS = 8000
            
            base_filename = metadata.get('filename', f'unknown_source_{metadata.get("internal_object_id", "")}')
            s3_friendly_name = "".join(c if c.isalnum() or c in ['-', '_'] else '_' for c in base_filename)
            s3_base_path = f"{source}/{s3_friendly_name}"
            
            complexity_metrics = {
                "text_density": len(markdown_text),
                "page_count": max(1, len(markdown_text) // MAX_CHARS)
            }
            
            adaptive_batch_size = self._calculate_adaptive_batch_size('structured', complexity_metrics)
            
            if len(markdown_text) <= MAX_CHARS:
                text_s3_key = f"{s3_base_path}/batch_1.txt"
                await self.s3_handler.upload_string(markdown_text, text_s3_key)
                
                batch_metadata = metadata.copy()
                batch_metadata.update({
                    "batch_number": 1,
                    "total_batches": 1,
                    "document_id": metadata.get("internal_object_id", "")
                })
                
                yield {
                    "batch": 1,
                    "text": markdown_text,
                    "metadata": batch_metadata
                }
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
                
                batches = []
                current_batch = []
                current_length = len(header)
                total_batches = 0
                
                for line in data_lines:
                    if current_length + len(line) > MAX_CHARS and current_batch:
                        batches.append(header + ''.join(current_batch))
                        current_batch = [line]
                        current_length = len(header) + len(line)
                        total_batches += 1
                    else:
                        current_batch.append(line)
                        current_length += len(line)
                
                if current_batch:
                    batches.append(header + ''.join(current_batch))
                    total_batches += 1
                
                for start_idx in range(0, len(batches), adaptive_batch_size):
                    end_idx = min(start_idx + adaptive_batch_size, len(batches))
                    upload_tasks = []
                    batch_results = []
                    
                    for i in range(start_idx, end_idx):
                        batch_text = batches[i]
                        batch_num = i + 1
                        
                        text_s3_key = f"{s3_base_path}/batch_{batch_num}.txt"
                        upload_task = self.s3_handler.upload_string(batch_text, text_s3_key)
                        upload_tasks.append(upload_task)
                        
                        batch_metadata = metadata.copy()
                        batch_metadata.update({
                            "batch_number": batch_num,
                            "total_batches": total_batches,
                            "document_id": metadata.get("internal_object_id", "")
                        })
                        
                        batch_results.append({
                            "batch": batch_num,
                            "text": batch_text,
                            "metadata": batch_metadata
                        })
                    
                    if upload_tasks:
                        await asyncio.gather(*upload_tasks)
                    
                    for result in batch_results:
                        yield result
        except Exception as e:
            logger.error(f"Error processing structured document: {e}")
            raise
    
    async def process_direct_document(self, file_data: bytes, file_type: str, source: str, metadata: Dict[str, Any]):
        try:
            loop = asyncio.get_running_loop()
            text = await loop.run_in_executor(None, lambda: file_data.decode('utf-8', errors='replace'))
            
            if file_type != 'markdown':
                markdown_text = await loop.run_in_executor(None, self.markdown.convert_text, text)
            else:
                markdown_text = text
            
            base_filename = metadata.get('filename', f'unknown_source_{metadata.get("internal_object_id", "")}')
            s3_friendly_name = "".join(c if c.isalnum() or c in ['-', '_'] else '_' for c in base_filename)
            s3_base_path = f"{source}/{s3_friendly_name}"
            
            text_s3_key = f"{s3_base_path}/content.txt"
            await self.s3_handler.upload_string(markdown_text, text_s3_key)
            
            document_metadata = metadata.copy()
            document_metadata.update({
                "document_id": metadata.get("internal_object_id", ""),
                "content_type": file_type
            })
            
            yield {
                "text": markdown_text,
                "metadata": document_metadata
            }
        except Exception as e:
            logger.error(f"Error processing direct document: {e}")
            raise

    async def process(self, data: bytes, metadata: Optional[Dict[str, Any]] = None, source: Optional[str] = None):
        if not data or not isinstance(data, bytes):
            error_msg = f"Invalid data type for file processing: {type(data)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        try:
            file_type = self.detect_file_type(data)
            category = self.categorize_file(file_type)
            
            if metadata is None:
                metadata = {}
            
            metadata["file_type"] = file_type
            
            if source is None:
                source = "unknown_source"
            
            if category == 'unstructured':
                async for item in self.process_unstructured_document(data, file_type, source, metadata):
                    yield item
            elif category == 'structured':
                async for item in self.process_structured_document(data, file_type, source, metadata):
                    yield item
            elif category == 'direct':
                async for item in self.process_direct_document(data, file_type, source, metadata):
                    yield item
            else:
                error_msg = f"Unsupported file type: {file_type}"
                logger.error(error_msg)
                raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Error processing file: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise

    def _analyze_document_complexity(self, pdf_reader, sample_size=3):
        total_pages = len(pdf_reader.pages)
        page_indices = [0]  # Always include first page
        
        if total_pages > 1:
            page_indices.append(total_pages // 2)
        if total_pages > 2:
            page_indices.append(total_pages - 1)
            
        import random
        while len(page_indices) < sample_size and len(page_indices) < total_pages:
            idx = random.randint(1, total_pages - 2)
            if idx not in page_indices:
                page_indices.append(idx)
        
        total_text_length = 0
        total_image_count = 0
        
        for idx in page_indices:
            page = pdf_reader.pages[idx]
            text = page.extract_text()
            total_text_length += len(text) if text else 0
            
            if '/Resources' in page and '/XObject' in page['/Resources']:
                total_image_count += len(page['/Resources']['/XObject'])
        
        avg_text_per_page = total_text_length / len(page_indices)
        avg_image_per_page = total_image_count / len(page_indices)
        normalized_image_density = min(1.0, avg_image_per_page / max(1, avg_image_per_page + 1))
        
        return {
            "page_count": total_pages,
            "avg_text_per_page": avg_text_per_page,
            "image_density": normalized_image_density,
        }
    
    def _calculate_adaptive_batch_size(self, document_type, complexity_metrics=None):
        default_size = self.default_batch_sizes.get(document_type, 4)
        if not complexity_metrics:
            return default_size
        
        batch_size = default_size
        
        page_count = complexity_metrics.get("page_count", 0)
        if page_count > self.complexity_thresholds["page_count"]["high"]:
            batch_size = max(1, batch_size - 2)
        elif page_count > self.complexity_thresholds["page_count"]["medium"]:
            batch_size = max(1, batch_size - 1)
        elif page_count < self.complexity_thresholds["page_count"]["low"]:
            batch_size += 1
        
        image_density = complexity_metrics.get("image_density", 0)
        if image_density > self.complexity_thresholds["image_density"]["high"]:
            batch_size = max(1, batch_size - 1)
        elif image_density < self.complexity_thresholds["image_density"]["low"]:
            batch_size += 1
        
        text_density = complexity_metrics.get("avg_text_per_page", 0)
        if text_density > self.complexity_thresholds["text_density"]["high"]:
            batch_size = max(1, batch_size - 1)
        
        cpu_count = os.cpu_count() or 4
        max_batch = cpu_count - 1
        batch_size = min(max_batch, batch_size)
        
        return max(1, batch_size)
    
    async def _prefetch_pages(self, pdf_reader, start_idx, end_idx, total_pages):
        try:
            if start_idx >= total_pages or end_idx <= start_idx:
                return
            
            end_idx = min(end_idx, total_pages)
                
            for page_num in range(start_idx, end_idx):
                writer = pypdf.PdfWriter()
                writer.add_page(pdf_reader.pages[page_num])
                single_page_pdf = io.BytesIO()
                writer.write(single_page_pdf)
                single_page_pdf.seek(0)
                page_data = single_page_pdf.getvalue()
                
                await self._prefetch_queue.put((page_num, page_data))
                
        except Exception as e:
            logger.error(f"Error during page pre-fetching: {e}")
    
    async def _get_prefetched_page(self, page_num, timeout=0.1):
        try:
            if not self._prefetch_queue.empty():
                queue_items = []
                found_page = None
                
                while not self._prefetch_queue.empty():
                    item = await asyncio.wait_for(self._prefetch_queue.get(), timeout)
                    if item[0] == page_num:
                        found_page = item
                    else:
                        queue_items.append(item)
                
                for item in queue_items:
                    await self._prefetch_queue.put(item)
                    
                return found_page
        except (asyncio.TimeoutError, asyncio.QueueEmpty):
            pass
        except Exception as e:
            logger.error(f"Error retrieving pre-fetched page: {e}")
            
        return None