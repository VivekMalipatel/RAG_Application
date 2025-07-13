import logging
from fastapi import APIRouter, UploadFile, File, Depends, Form, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional
import json
import boto3
from core.db.database import get_db
from schemas.schemas import UrlIngestRequest, RawTextIngestRequest, IngestResponse
from core.queue.rabbitmq_handler import rabbitmq_handler
from pydantic import BaseModel
from typing import Optional
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

class IngestRequest(BaseModel):
    user_id: str
    file_id: str
    filename: str
    s3_url: str
    metadata: Optional[dict] = None

class IngestResponse(BaseModel):
    collection_id: str
    document_id: str
    status: str

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/file")
async def ingest_file_process(
    file: UploadFile = File(None),
    metadata: str = Form(...),
    db: AsyncSession = Depends(get_db)
):
    metadata_dict = json.loads(metadata)
    if file:
        file_content = await file.read()
        logger.info(f"Received file: {file.filename}, size: {len(file_content)} bytes, metadata: {metadata_dict}")
    elif "s3_path" in metadata_dict:
        # Download from S3
        s3 = boto3.client("s3")
        bucket, key = parse_s3_path(metadata_dict["s3_path"])
        s3_obj = s3.get_object(Bucket=bucket, Key=key)
        file_content = s3_obj["Body"].read()
    else:
        raise HTTPException(status_code=400, detail="No file or s3_path provided")

    # Enqueue file processing asynchronously (fire and forget)
    async def enqueue_file_task():
        await rabbitmq_handler.enqueue_file(
            file_content=file_content,
            filename=file.filename if file else key.split("/")[-1],
            source=metadata_dict.get("source", "external"),
            metadata=metadata_dict
        )
    import asyncio
    asyncio.create_task(enqueue_file_task())

    # Always return in OpenWebUI preview format
    return {
        "page_content": file_content.decode("utf-8", errors="replace"),
        "metadata": metadata_dict
    }

@router.post("/url", response_model=IngestResponse)
async def ingest_url(
    url_request: UrlIngestRequest,
    db: AsyncSession = Depends(get_db)
):
    logger.info(f"Received URL: {url_request.url} from source: {url_request.source}")
    
    queue_id = await rabbitmq_handler.enqueue_url(
        url=url_request.url,
        source=url_request.source,
        metadata=url_request.metadata
    )
    
    return IngestResponse(
        collection_id="unknown",
        document_id=queue_id,
        status="queued"
    )

@router.post("/raw-text", response_model=IngestResponse)
async def ingest_raw_text(
    text_request: RawTextIngestRequest,
    db: AsyncSession = Depends(get_db)
):
    logger.info(f"Received raw text from source: {text_request.source}")
    
    queue_id = await rabbitmq_handler.enqueue_text(
        text=text_request.text,
        source=text_request.source,
        metadata=text_request.metadata
    )
    
    return IngestResponse(
        collection_id="unknown",
        document_id=queue_id,
        status="queued"
    )

def parse_s3_path(s3_path: str):
    """Parse an S3 path like s3://bucket/key/to/file.ext into (bucket, key)"""
    if s3_path.startswith("s3://"):
        s3_path = s3_path[5:]
    parts = s3_path.split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid S3 path: {s3_path}")
    return parts[0], parts[1]