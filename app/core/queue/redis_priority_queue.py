import json
import logging
from typing import Optional, Dict
from app.core.cache.session_redis import redis_session
from app.config import settings

class RedisPriorityQueue:
    """Manages priority queue operations with Redis BRPOP pattern"""
    
    def __init__(self):
        self.redis = redis_session.client
        self.queue_map = {
            "chat": settings.REDIS_CHAT_QUEUE,
            "standard": settings.REDIS_STANDARD_QUEUE
        }

    async def push_to_queue(self, queue_type: str, data: dict) -> None:
        """Push event to specified queue with validation"""
        if queue_type not in self.queue_map:
            raise ValueError(f"Invalid queue type: {queue_type}. Valid types: {list(self.queue_map.keys())}")
        
        queue_name = self.queue_map[queue_type]
        try:
            await self.redis.lpush(queue_name, json.dumps(data))
            logging.debug(f"Event pushed to {queue_name}: {data['event_id']}")
        except Exception as e:
            logging.error(f"Failed to push to {queue_name}: {str(e)}")
            raise

    async def consume_events(self) -> Optional[Dict]:
        """BRPOP implementation for priority consumption"""
        try:
            result = await self.redis.brpop(
                [self.queue_map["chat"], self.queue_map["standard"]], 
                timeout=30
            )
            if result:
                return json.loads(result[1])
        except Exception as e:
            logging.error(f"Queue consumption failed: {str(e)}")
        return None
