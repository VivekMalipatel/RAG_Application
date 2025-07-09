from langgraph.store.redis import AsyncRedisStore
from langgraph.store.base import IndexConfig, TTLConfig
from redis.asyncio import Redis as AsyncRedis
from typing import Optional, Any

class BaseMemoryStore(AsyncRedisStore):

    def __init__(
        self,
        redis_url: Optional[str] = None,
        *,
        redis_client: Optional[AsyncRedis] = None,
        index: Optional[IndexConfig] = None,
        connection_args: Optional[dict[str, Any]] = None,
        ttl: Optional[dict[str, Any]] = None,
        cluster_mode: Optional[bool] = None,
    ) -> None:

        super().__init__(redis_url=redis_url, redis_client=redis_client, index=index, connection_args=connection_args, ttl=ttl, cluster_mode=cluster_mode)