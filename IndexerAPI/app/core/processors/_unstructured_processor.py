import io
import os
import asyncio
import base64
import logging
from typing import Dict, Any, List

import pypdf

from core.processors.base_processor import BaseProcessor
from core.processors.utils import rasterize_and_encode, convert_to_pdf, detect_file_type
from core.markitdown.markdown_handler import MarkDown
from core.model.model_handler import get_global_model_handler
from core.storage.s3_handler import get_global_s3_handler

logger = logging.getLogger(__name__)

class UnstructuredProcessor(BaseProcessor):
    
    def __init__(self):
        logger.info("Initializing UnstructuredProcessor")
        self.markdown = MarkDown()
        self.model_handler = get_global_model_handler()
        logger.info("UnstructuredProcessor initialized successfully")

    async def process(self, task_message) -> Dict[str, Any]:
        raise NotImplementedError("Use process_unstructured_document method instead")

    async def process_single_page(self, page_data: bytes, page_num: int, s3_base_path: str) -> Dict[str, Any]:
        logger.debug(f"Starting processing for page {page_num + 1} with {len(page_data)} bytes of data")
        try:
            s3_handler = await get_global_s3_handler()
            
            loop = asyncio.get_running_loop()
            logger.debug(f"Extracting text and generating image for page {page_num + 1}")
            extracted_text_task = loop.run_in_executor(None, self.markdown.convert_bytes, page_data)
            image_task = loop.run_in_executor(None, rasterize_and_encode, page_data, page_num)
            
            extracted_text, (_, image_base64) = await asyncio.gather(extracted_text_task, image_task)
            logger.debug(f"Page {page_num + 1}: extracted {len(extracted_text)} characters of text and generated base64 image")
            
            messages = [
                {
                    "role": "user", 
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                        {"type": "text", "text": f"Extracted text from page: {extracted_text}"}
                    ]
                }
            ]

            logger.debug(f"Extracting entities and relationships for page {page_num + 1}")
            entities_relationships_task = self.model_handler.extract_entities_relationships(messages)

            embed_messages = [
                {
                    "role": "user", 
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                    ]
                }
            ]
            logger.debug(f"Generating page embedding for page {page_num + 1}")
            page_embedding_task = self.model_handler.embed(embed_messages)
            
            entities_relationships, page_embedding = await asyncio.gather(
                entities_relationships_task,
                page_embedding_task
            )
            logger.debug(f"Page {page_num + 1}: completed entity extraction and embedding generation")
            
            entities, relationships = await self.model_handler.embed_entity_relationship_profiles(
                entities_relationships["entities"], 
                entities_relationships["relationships"]
            )
            logger.debug(f"Page {page_num + 1}: found {len(entities)} entities and {len(relationships)} relationships")
            
            logger.debug(f"Uploading image to S3 for page {page_num + 1}")
            image_bytes_data = base64.b64decode(image_base64)
            image_s3_key = f"metadata/{s3_base_path}/page_{page_num + 1}.jpg"
            await s3_handler.upload_bytes(image_bytes_data, image_s3_key)
            image_s3_url = f"{s3_handler.endpoint_url}/{s3_handler.bucket_name}/{image_s3_key}"
            logger.debug(f"Page {page_num + 1}: image uploaded to {image_s3_url}")
            
            logger.info(f"Successfully processed page {page_num + 1}")
            return {
                "page_number": page_num + 1,
                "messages": messages,
                "entities": entities,
                "relationships": relationships,
                "image_s3_url": image_s3_url,
                "embedding": page_embedding[0] if page_embedding else None
            }
        except Exception as e:
            logger.error(f"Error processing page {page_num + 1}: {e}")
            return {
                "page_number": page_num + 1,
                "messages": f"Error processing page {page_num + 1}: {str(e)}",
                "entities": [],
                "relationships": [],
                "image_s3_url": "",
                "embedding": None
            }

    async def convert_to_images(self, file_data: bytes, s3_base_path: str) -> List[Dict[str, Any]]:
        logger.info(f"Starting PDF conversion to images with {len(file_data)} bytes of data")
        try:
            pdf_stream = io.BytesIO(file_data)
            pdf_reader = pypdf.PdfReader(pdf_stream)
            total_pages = len(pdf_reader.pages)
            logger.info(f"Processing PDF with {total_pages} pages")
    
            logger.info(f"Creating processing tasks for {total_pages} pages")
            page_tasks = []
            for page_num in range(total_pages):
                logger.debug(f"Preparing page {page_num + 1}/{total_pages} for processing")
                writer = pypdf.PdfWriter()
                writer.add_page(pdf_reader.pages[page_num])
                single_page_pdf = io.BytesIO()
                writer.write(single_page_pdf)
                single_page_pdf.seek(0)
                
                task = self.process_single_page(single_page_pdf.getvalue(), page_num, s3_base_path)
                page_tasks.append(task)
            
            logger.info(f"Executing parallel processing for {len(page_tasks)} pages")
            result = await asyncio.gather(*page_tasks)
            logger.info(f"Successfully processed all {len(result)} pages")
            return result
                
        except Exception as e:
            logger.error(f"Error converting file to images: {e}")
            return [{
                "page_number": 1,
                "messages": f"Error converting file: {str(e)}",
                "entities": [],
                "relationships": [],
                "image_s3_url": "",
                "embedding": None
            }]

    async def process_unstructured_document(self, file_data: bytes, file_type: str, s3_base_path: str) -> Dict[str, Any]:
        logger.info(f"Starting unstructured document processing for file type '{file_type}' with {len(file_data)} bytes")
        
        logger.info(f"Converting '{file_type}' file to PDF format")
        file_data = await convert_to_pdf(file_data, file_type)
        logger.debug(f"File converted to PDF, new size: {len(file_data)} bytes")

        logger.debug("Detecting converted file type to verify PDF conversion")
        detected_type = await detect_file_type(file_data)
        logger.debug(f"Detected file type after conversion: {detected_type}")
        
        if detected_type != 'pdf':
            error_msg = f"File conversion to PDF failed: {file_type}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info("PDF conversion verified, starting image processing")
        result = await self.convert_to_images(file_data, s3_base_path)
        
        logger.info(f"Unstructured document processing completed with {len(result['data']) if 'data' in str(result) else len(result)} pages")
        return {
            "data": result,
            "category": "unstructured"
        }
