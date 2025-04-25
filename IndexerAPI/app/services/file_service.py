import logging
import uuid
import os
from fastapi import UploadFile, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import insert

from app.models.file_item import FileItem
from app.schemas.file_schemas import FileIngestRequest
from app.utils.file_utils import save_upload_file
from app.processors.file_processor import FileProcessor

logger = logging.getLogger(__name__)

class FileService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.storage_dir = os.environ.get("FILE_STORAGE_DIR", "temp_storage")
        
        # Create storage directory if it doesn't exist
        os.makedirs(self.storage_dir, exist_ok=True)
    
    async def process_file(self, file: UploadFile, request: FileIngestRequest, background_tasks: BackgroundTasks):
        """Process uploaded file and queue it for processing"""
        # Generate a unique ID for this file
        file_id = str(uuid.uuid4())
        
        # Determine file type from file extension
        file_type = self._get_file_type(file.filename)
        
        # Save file to temporary storage
        storage_path = await save_upload_file(file, self.storage_dir, file_id)
        
        # Create file item record
        file_item = FileItem(
            id=file_id,
            filename=file.filename,
            file_type=file_type,
            source=request.source,
            metadata=request.metadata,
            status="queued",
            storage_path=storage_path
        )
        
        # Save to database
        self.db.add(file_item)
        await self.db.commit()
        
        # Queue for background processing
        background_tasks.add_task(self._process_in_background, file_id, storage_path, file_type)
        
        return file_item
    
    async def _process_in_background(self, file_id: str, storage_path: str, file_type: str):
        """Process file in background"""
        # Update status to processing
        file_item = await self.db.get(FileItem, file_id)
        file_item.status = "processing"
        await self.db.commit()
        
        try:
            # Get appropriate processor based on file type
            processor = FileProcessor.get_processor(file_type)
            
            # Process file
            result = await processor.process(storage_path)
            
            # Update file item with processing result
            file_item.status = "completed"
            file_item.embedding_stored = result.get("embedding_id")
            await self.db.commit()
            
        except Exception as e:
            logger.error(f"Error processing file {file_id}: {str(e)}")
            file_item.status = "failed"
            file_item.status_message = str(e)
            await self.db.commit()
    
    def _get_file_type(self, filename: str) -> str:
        """Determine file type from filename"""
        if not filename:
            return "unknown"
            
        ext = filename.split(".")[-1].lower()
        
        # Map file extensions to general types
        extension_map = {
            "pdf": "pdf",
            "docx": "document",
            "doc": "document",
            "xlsx": "spreadsheet",
            "xls": "spreadsheet",
            "csv": "csv",
            "txt": "text",
            "md": "markdown",
            "jpg": "image",
            "jpeg": "image",
            "png": "image",
            "mp4": "video",
            "mov": "video",
            "mp3": "audio",
            "wav": "audio",
            "json": "json",
            "html": "html",
        }
        
        return extension_map.get(ext, "unknown")