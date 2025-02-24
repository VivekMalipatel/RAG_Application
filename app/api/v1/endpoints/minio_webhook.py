from fastapi import APIRouter, HTTPException, status, Request
import logging
import os
from uuid import uuid4
from typing import List
from app.core.queue.redis_priority_queue import RedisPriorityQueue
from app.config import settings
import asyncio

router = APIRouter()

@router.post("/webhook/", status_code=status.HTTP_202_ACCEPTED)
async def minio_webhook(request: Request):
    """Handle MinIO bucket notifications, only processing file-related events."""
    try:
        # Parse incoming webhook JSON
        event = await request.json()
        logging.info(f"Received MinIO Webhook Event: {event}")

        # Extract file key
        file_key = event.get("Key", "")
        path_parts = file_key.split("/")
        event_name = event.get("EventName", "")

        if not file_key or not event_name:
            logging.error("Invalid payload received")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing required event fields"
            )
        
        # Simulate processing delay
        await asyncio.sleep(2)

        # Validate that the key belongs to a file inside "standard/" or "chat/"
        if "." not in path_parts[-1]:
            logging.info(f"Ignoring non-relevant object: {file_key}")
            return {"status": "ignored", "reason": "Not a file event"}

        # Event Filtering (only process object creation & deletion events)
        if not event_name.startswith('s3:ObjectCreated:CompleteMultipartUpload') and \
           not event_name.startswith('s3:ObjectRemoved:Delete'):
            logging.info(f"Ignoring non-relevant event: {event_name}")
            return {"status": "ignored", "reason": "Non-file operation event"}

        # Extract user ID & upload type from path
        user_id = path_parts[1]  # Assuming user_id is after bucket name
        upload_type = "chat" if "chat" in path_parts else "standard"

        # Queue Selection and Task Queuing
        queue_name = settings.REDIS_CHAT_QUEUE if upload_type == "chat" else settings.REDIS_STANDARD_QUEUE

        task_data = {
            "event_id": str(uuid4()),
            "user_id": user_id,
            "path": file_key,
            "event_type": event_name,
            "upload_channel": upload_type,
            "timestamp": event.get("EventTime")
        }

        queue = RedisPriorityQueue()
        await queue.push_to_queue(
            queue_type="chat" if upload_type == "chat" else "standard",
            data=task_data
        )

        logging.info(f"Queued {upload_type} upload successfully: {file_key}")
        
        return {
            "success": True,
            "queue": queue_name,
            "priority": "high" if upload_type == "chat" else "normal"
        }

    except ValueError as ve:
        logging.warning(f"Path validation failed: {str(ve)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )

    except Exception as e:
        logging.error(f"Webhook processing failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal processing error"
        )