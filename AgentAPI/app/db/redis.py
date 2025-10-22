from redis.asyncio import Redis as AsyncRedis
from typing import Optional
from config import config

class Redis:
    _instance: Optional['Redis'] = None
    _redis_session: Optional[AsyncRedis] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def init_session(self):
        if self._redis_session is None:
            self._redis_session = AsyncRedis.from_url(
                config.REDIS_URI,
                decode_responses=True
            )

    def get_session(self) -> AsyncRedis:
        if self._redis_session is None:
            self.init_session()
        return self._redis_session

    def close_session(self):
        if self._redis_session:
            self._redis_session.close()
            self._redis_session = None

redis = Redis()