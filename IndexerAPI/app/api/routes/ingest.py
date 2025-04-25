import logging
from fastapi import APIRouter, UploadFile, File, Depends, BackgroundTasks, Form, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, Dict, Any
import json

from app.db.database import get_db
from app.schemas.file_schemas import FileIngestRequest, UrlIngestRequest, RawTextIngestRequest, IngestResponse
from app.services.file_service import FileService
from app.services.url_service import UrlService
from app.services.text_service import TextService

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/file", response_model=IngestResponse)
async def ingest_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    source: str = Form(...),
    metadata: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db)
):
    """
    Upload a file for processing and indexing
    """
    logger.info(f"Received file: {file.filename} from source: {source}")
    
    # Parse metadata if provided
    meta_dict = None
    if metadata:
        try:
            meta_dict = json.loads(metadata)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid metadata format")
    
    # Create file request
    file_request = FileIngestRequest(
        source=source,
        metadata=meta_dict
    )
    
    # Process the file using FileService
    file_service = FileService(db)
    result = await file_service.process_file(file, file_request, background_tasks)
    
    return IngestResponse(
        id=result.id,
        message="File accepted for processing"
    )

@router.post("/url", response_model=IngestResponse)
async def ingest_url(
    url_request: UrlIngestRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """
    Submit a URL for processing and indexing
    """
    logger.info(f"Received URL: {url_request.url} from source: {url_request.source}")
    
    # Process the URL using UrlService
    url_service = UrlService(db)
    result = await url_service.process_url(url_request, background_tasks)
    
    return IngestResponse(
        id=result.id,
        message="URL accepted for processing"
    )

@router.post("/raw-text", response_model=IngestResponse)
async def ingest_raw_text(
    text_request: RawTextIngestRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """
    Submit raw text for processing and indexing
    """
    logger.info(f"Received raw text from source: {text_request.source}")
    
    # Process the text using TextService
    text_service = TextService(db)
    result = await text_service.process_text(text_request, background_tasks)
    
    return IngestResponse(
        id=result.id,
        message="Raw text accepted for processing"
    )