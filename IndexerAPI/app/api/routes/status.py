import logging
from fastapi import APIRouter

from core.queue.rabbitmq_handler import rabbitmq_handler

router = APIRouter()
logger = logging.getLogger(__name__)


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