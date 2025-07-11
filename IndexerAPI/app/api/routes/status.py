import logging
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.schemas.schemas import StatusResponse
from app.queue.rabbitmq_handler import rabbitmq_handler

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/queue/info")
async def get_queue_info():
    """Get RabbitMQ queue information"""
    logger.info("Getting RabbitMQ queue information")
    
    try:
        queue_info = await rabbitmq_handler.get_queue_info()
        return queue_info
    except Exception as e:
        logger.error(f"Error getting queue info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting queue info: {str(e)}")

@router.post("/queue/purge")
async def purge_queue():
    """Purge all messages from the queue"""
    logger.info("Purging RabbitMQ queue")
    
    try:
        await rabbitmq_handler.purge_queue()
        return {"message": "Queue purged successfully"}
    except Exception as e:
        logger.error(f"Error purging queue: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error purging queue: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        queue_info = await rabbitmq_handler.get_queue_info()
        return {
            "status": "healthy",
            "rabbitmq_connected": queue_info["is_connected"],
            "queue_info": queue_info
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "rabbitmq_connected": False
        }