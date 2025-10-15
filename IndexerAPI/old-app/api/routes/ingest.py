import logging
import uuid
from fastapi import APIRouter, HTTPException
from schemas.schemas import FileIngestRequest, UrlIngestRequest, RawTextIngestRequest, IngestResponse
from core.queue.rabbitmq_handler import rabbitmq_handler
from core.queue.task_types import TaskMessage, TaskType

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/file", response_model=IngestResponse)
async def ingest_file(
    request: FileIngestRequest,
):
    logger.info(f"Received file ingest request for s3_url: {request.s3_url}")
    
    try:
        task_id = str(uuid.uuid1())
        task_message = TaskMessage(
            task_id=task_id,
            task_type=TaskType.FILE,
            payload=request.model_dump()
        )
        
        await rabbitmq_handler.enqueue_task(task_message)
        
        return IngestResponse(
            task_id=task_id
        )
        
    except Exception as e:
        logger.error(f"Error processing file ingestion: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")

@router.post("/url", response_model=IngestResponse)
async def ingest_url(
    request: UrlIngestRequest,
):
    logger.info(f"Received URL ingest request for: {request.url}")
    
    try:
        task_id = str(uuid.uuid1())
        task_message = TaskMessage(
            task_id=task_id,
            task_type=TaskType.URL,
            payload=request.model_dump()
        )
        
        await rabbitmq_handler.enqueue_task(task_message)
        
        return IngestResponse(
            task_id=task_id
        )
        
    except Exception as e:
        logger.error(f"Error processing URL ingestion: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process URL: {str(e)}")

@router.post("/raw-text", response_model=IngestResponse)
async def ingest_raw_text(
    request: RawTextIngestRequest,
):
    logger.info(f"Received raw text ingest request from source: {request.source}")
    
    try:
        task_id = str(uuid.uuid1())
        task_message = TaskMessage(
            task_id=task_id,
            task_type=TaskType.TEXT,
            payload=request.model_dump()
        )
        
        await rabbitmq_handler.enqueue_task(task_message)
        
        return IngestResponse(
            task_id=task_id
        )
        
    except Exception as e:
        logger.error(f"Error processing raw text ingestion: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process raw text: {str(e)}")