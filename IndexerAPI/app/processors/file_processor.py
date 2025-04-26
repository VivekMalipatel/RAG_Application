import logging
import os
import uuid
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, BinaryIO

import fitz  # PyMuPDF for PDF processing
from PIL import Image

from app.processors.input_type_detector import InputTypeDetector
from app.core.model.embedding import EmbeddingGenerator
from app.core.model.text_generation import TextGenerator
from app.core.markitdown.converter import MarkdownConverter

logger = logging.getLogger(__name__)

class FileProcessor:
    
    def __init__(
        self,
        temp_dir: str = "temp",
        embedding_generator: Optional[EmbeddingGenerator] = None,
        text_generator: Optional[TextGenerator] = None
    ):
        self.temp_dir = temp_dir
        self.embedding_generator = embedding_generator or EmbeddingGenerator()
        self.text_generator = text_generator or TextGenerator()
        self.input_type_detector = InputTypeDetector()
        self.markdown_converter = MarkdownConverter()
        
        os.makedirs(self.temp_dir, exist_ok=True)
        
        logger.info("File Processor initialized")
    
    async def process_file(
        self,
        file_path: str,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        logger.info(f"Processing file: {file_path}")
        
        try:
            job_id = str(uuid.uuid4())
            
            file_type = self.input_type_detector.detect_file_type(file_path)
            logger.info(f"Detected file type: {file_type}")
            
            processing_metadata = {
                "id": job_id,
                "file_path": file_path,
                "file_type": file_type,
                "original_metadata": metadata or {},
                "timestamp": datetime.now().isoformat(),
                "status": "processing"
            }
            
            if file_type == "pdf":
                result = await self.process_pdf(file_path, processing_metadata)
            elif file_type == "document":
                result = await self.process_document(file_path, processing_metadata)
            elif file_type == "image":
                result = await self.process_image(file_path, processing_metadata)
            elif file_type == "text" or file_type == "markdown":
                result = await self.process_text_file(file_path, processing_metadata)
            elif file_type == "spreadsheet" or file_type == "csv":
                result = await self.process_spreadsheet(file_path, processing_metadata)
            elif file_type == "audio":
                result = {"success": False, "error": "Audio processing not yet implemented"}
            elif file_type == "video":
                result = {"success": False, "error": "Video processing not yet implemented"}
            else:
                result = {"success": False, "error": f"Unsupported file type: {file_type}"}
            
            processing_metadata["status"] = "completed" if result.get("success", False) else "failed"
            if not result.get("success", False):
                processing_metadata["error"] = result.get("error", "Unknown error")
            
            return {
                **result,
                "metadata": processing_metadata
            }
            
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def process_pdf(
        self,
        file_path: str,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        logger.info(f"Processing PDF: {file_path}")
        
        try:
            job_dir = os.path.join(self.temp_dir, metadata["id"])
            os.makedirs(job_dir, exist_ok=True)
            
            doc = fitz.open(file_path)
            
            full_text = ""
            page_texts = []
            
            image_paths = []
            page_image_paths = []
            
            for page_num, page in enumerate(doc):
                page_text = page.get_text()
                page_texts.append(page_text)
                full_text += page_text
                
                page_image_path = os.path.join(job_dir, f"page_{page_num}.png")
                pixmap = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
                pixmap.save(page_image_path)
                page_image_paths.append(page_image_path)
                
                image_list = page.get_images(full=True)
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_data = base_image["image"]
                    
                    img_filename = f"image_p{page_num}_i{img_index}.png"
                    img_path = os.path.join(job_dir, img_filename)
                    
                    with open(img_path, "wb") as img_file:
                        img_file.write(image_data)
                    
                    image_paths.append(img_path)
            
            image_captions = []
            for img_path in image_paths:
                caption = await self._generate_image_caption(img_path)
                image_captions.append({
                    "path": img_path,
                    "caption": caption
                })
            
            markdown = await self.markdown_converter.convert_text(full_text)
            
            embedding_result = await self.embedding_generator.generate_embedding(markdown)
            
            if not embedding_result.get("success", False):
                logger.error(f"Error generating embedding: {embedding_result.get('error', 'Unknown error')}")
            
            result = {
                "success": True,
                "text": full_text,
                "markdown": markdown,
                "pages": len(doc),
                "page_texts": page_texts,
                "page_image_paths": page_image_paths,
                "extracted_images": image_paths,
                "image_captions": image_captions,
                "embedding": embedding_result.get("embedding") if embedding_result.get("success", False) else None,
                "temp_dir": job_dir
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def process_document(
        self,
        file_path: str,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        logger.info(f"Processing document: {file_path}")
        
        return {
            "success": False,
            "error": "Document processing not fully implemented yet"
        }
    
    async def process_image(
        self,
        file_path: str,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        logger.info(f"Processing image: {file_path}")
        
        try:
            job_dir = os.path.join(self.temp_dir, metadata["id"])
            os.makedirs(job_dir, exist_ok=True)
            
            image_path = os.path.join(job_dir, os.path.basename(file_path))
            shutil.copy(file_path, image_path)
            
            caption = await self._generate_image_caption(image_path)
            
            markdown = await self.markdown_converter.convert_text(caption)
            
            embedding_result = await self.embedding_generator.generate_embedding(markdown)
            
            if not embedding_result.get("success", False):
                logger.error(f"Error generating embedding: {embedding_result.get('error', 'Unknown error')}")
            
            result = {
                "success": True,
                "image_path": image_path,
                "caption": caption,
                "markdown": markdown,
                "embedding": embedding_result.get("embedding") if embedding_result.get("success", False) else None,
                "temp_dir": job_dir
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def process_text_file(
        self,
        file_path: str,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        logger.info(f"Processing text file: {file_path}")
        
        try:
            job_dir = os.path.join(self.temp_dir, metadata["id"])
            os.makedirs(job_dir, exist_ok=True)
            
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            
            markdown = await self.markdown_converter.convert_text(text)
            
            embedding_result = await self.embedding_generator.generate_embedding(markdown)
            
            if not embedding_result.get("success", False):
                logger.error(f"Error generating embedding: {embedding_result.get('error', 'Unknown error')}")
            
            result = {
                "success": True,
                "text": text,
                "markdown": markdown,
                "embedding": embedding_result.get("embedding") if embedding_result.get("success", False) else None,
                "temp_dir": job_dir
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing text file: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def process_spreadsheet(
        self,
        file_path: str,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        logger.info(f"Processing spreadsheet: {file_path}")
        
        return {
            "success": False,
            "error": "Spreadsheet processing not fully implemented yet"
        }
    
    async def _generate_image_caption(self, image_path: str) -> str:
        try:
            with Image.open(image_path) as img:
                max_size = 1024
                if max(img.size) > max_size:
                    ratio = max_size / max(img.size)
                    new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                    img = img.resize(new_size, Image.LANCZOS)
            
            caption = f"Image file: {os.path.basename(image_path)}"
            
            return caption
            
        except Exception as e:
            logger.error(f"Error generating image caption: {str(e)}")
            return "Image caption generation failed"