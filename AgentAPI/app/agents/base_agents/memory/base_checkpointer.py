from langgraph.checkpoint.redis import AsyncRedisSaver
from redis.asyncio import Redis as AsyncRedis
from redis.asyncio.cluster import RedisCluster as AsyncRedisCluster
from config import config
from typing import Optional, Union, Dict, Any, AsyncIterator
from langgraph.checkpoint.base import CheckpointTuple
from langgraph.types import RunnableConfig
import asyncio
import logging

class BaseMemorySaver(AsyncRedisSaver):

    def __init__(
        self,
        redis_url: Optional[str] = None,
        *,
        redis_client: Optional[Union[AsyncRedis, AsyncRedisCluster]] = None,
        connection_args: Optional[Dict[str, Any]] = None,
        ttl: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            redis_url=redis_url,
            redis_client=redis_client,
            connection_args=connection_args,
            ttl=ttl,
        )
    


