import logging
from fastapi import APIRouter, UploadFile, File, Depends, Form, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional
import json

from app.db.database import get_db
from app.schemas.schemas import FileIngestRequest, UrlIngestRequest, RawTextIngestRequest, IngestResponse
from app.queue import QueueHandler

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/file", response_model=IngestResponse)
async def ingest_file(
    file: UploadFile = File(...),
    source: str = Form(...),
    metadata: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db)
):
    logger.info(f"Received file: {file.filename} from source: {source}")
    
    meta_dict = None
    if metadata:
        try:
            meta_dict = json.loads(metadata)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid metadata format")
    
    # Read file content
    file_content = await file.read()
    
    # Add file to queue
    queue_handler = QueueHandler(db)
    queue_id = await queue_handler.enqueue_file(
        file_content=file_content,
        filename=file.filename,
        source=source,
        metadata=meta_dict
    )
    
    return IngestResponse(
        id=queue_id,
        message="File accepted for processing"
    )

@router.post("/url", response_model=IngestResponse)
async def ingest_url(
    url_request: UrlIngestRequest,
    db: AsyncSession = Depends(get_db)
):
    logger.info(f"Received URL: {url_request.url} from source: {url_request.source}")
    
    # Add URL to queue
    queue_handler = QueueHandler(db)
    queue_id = await queue_handler.enqueue_url(
        url=url_request.url,
        source=url_request.source,
        metadata=url_request.metadata
    )
    
    return IngestResponse(
        id=queue_id,
        message="URL accepted for processing"
    )

@router.post("/raw-text", response_model=IngestResponse)
async def ingest_raw_text(
    text_request: RawTextIngestRequest,
    db: AsyncSession = Depends(get_db)
):
    logger.info(f"Received raw text from source: {text_request.source}")
    
    # Add text to queue
    queue_handler = QueueHandler(db)
    queue_id = await queue_handler.enqueue_text(
        text=text_request.text,
        source=text_request.source,
        metadata=text_request.metadata
    )
    
    return IngestResponse(
        id=queue_id,
        message="Raw text accepted for processing"
    )