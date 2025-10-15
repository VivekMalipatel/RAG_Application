import logging
import uuid
from urllib.parse import unquote, urlparse
from fastapi import APIRouter, HTTPException
from schemas.schemas import FileIngestRequest, UrlIngestRequest, RawTextIngestRequest, IngestResponse
from core.queue.rabbitmq_handler import rabbitmq_handler
from core.queue.task_types import TaskMessage, TaskType

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/file", response_model=IngestResponse)
async def ingest_file(request: FileIngestRequest):
    logger.info(f"Received file ingest request for s3_url: {request.s3_url}")
    try:
        task_id = str(uuid.uuid1())
        payload = request.model_dump()
        metadata = payload.setdefault("metadata", {})
        parsed = urlparse(payload["s3_url"])
        s3_path = parsed.path or ""
        s3_filename = unquote(s3_path.rsplit("/", 1)[-1]) if s3_path else ""
        if s3_filename:
            metadata["filename"] = s3_filename
        task_message = TaskMessage(task_id=task_id, task_type=TaskType.FILE, payload=payload)
        await rabbitmq_handler.enqueue_task(task_message)
        return IngestResponse(task_id=task_id)
    except Exception as exc:
        logger.error(f"Error processing file ingestion: {exc}")
        raise HTTPException(status_code=500, detail=f"Failed to process file: {exc}")

@router.post("/url", response_model=IngestResponse)
async def ingest_url(request: UrlIngestRequest):
    logger.info(f"Received URL ingest request for: {request.url}")
    try:
        task_id = str(uuid.uuid1())
        task_message = TaskMessage(task_id=task_id, task_type=TaskType.URL, payload=request.model_dump())
        await rabbitmq_handler.enqueue_task(task_message)
        return IngestResponse(task_id=task_id)
    except Exception as exc:
        logger.error(f"Error processing URL ingestion: {exc}")
        raise HTTPException(status_code=500, detail=f"Failed to process URL: {exc}")

@router.post("/raw-text", response_model=IngestResponse)
async def ingest_raw_text(request: RawTextIngestRequest):
    logger.info(f"Received raw text ingest request from source: {request.source}")
    try:
        task_id = str(uuid.uuid1())
        task_message = TaskMessage(task_id=task_id, task_type=TaskType.TEXT, payload=request.model_dump())
        await rabbitmq_handler.enqueue_task(task_message)
        return IngestResponse(task_id=task_id)
    except Exception as exc:
        logger.error(f"Error processing raw text ingestion: {exc}")
        raise HTTPException(status_code=500, detail=f"Failed to process raw text: {exc}")
