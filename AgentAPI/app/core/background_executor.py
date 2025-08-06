import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Callable, Any
import redis.asyncio as async_redis
from config import config as envconfig

logger = logging.getLogger(__name__)

class BackgroundTaskManager:
    _instance: Optional['BackgroundTaskManager'] = None
    
    def __init__(self, max_workers: int = envconfig.BACKGROUND_TASK_MAX_WORKERS):
        self._executor: Optional[ThreadPoolExecutor] = None
        self._max_workers = max_workers
        self._is_initialized = False
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def initialize(self):
        if not self._is_initialized:
            self._executor = ThreadPoolExecutor(
                max_workers=self._max_workers, 
                thread_name_prefix="background_task"
            )
            self._is_initialized = True
            logger.info(f"Background task manager initialized with {self._max_workers} workers")
    
    def shutdown(self):
        if self._executor and self._is_initialized:
            self._executor.shutdown(wait=True)
            self._executor = None
            self._is_initialized = False
            logger.info("Background task manager shutdown completed")
    
    def submit_task(self, task: Callable, *args, inject_redis: bool = False, **kwargs):
        if not self._is_initialized or not self._executor:
            raise RuntimeError("Background task manager not initialized")
        
        def run_in_new_loop():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            redis_client = None
            try:
                if inject_redis:
                    redis_client = async_redis.Redis(
                        host=envconfig.REDIS_HOST,
                        port=envconfig.REDIS_PORT,
                        decode_responses=True
                    )
                    if asyncio.iscoroutinefunction(task):
                        return loop.run_until_complete(task(redis_client, *args, **kwargs))
                    else:
                        return task(redis_client, *args, **kwargs)
                else:
                    if asyncio.iscoroutinefunction(task):
                        return loop.run_until_complete(task(*args, **kwargs))
                    else:
                        return task(*args, **kwargs)
            finally:
                if redis_client:
                    loop.run_until_complete(redis_client.aclose())
                loop.close()
        
        future = self._executor.submit(run_in_new_loop)
        logger.debug(f"Submitted {'coroutine' if asyncio.iscoroutinefunction(task) else 'function'} {task.__name__} to background executor")
        return future
    
    @property
    def is_initialized(self) -> bool:
        return self._is_initialized

background_manager = BackgroundTaskManager()

def submit_background_task(task: Callable, *args, **kwargs):
    try:
        return background_manager.submit_task(task, *args, **kwargs)
    except RuntimeError as e:
        logger.warning(f"Background task submission failed: {e}")
        return None

def submit_background_task_with_redis(task: Callable, *args, **kwargs):
    try:
        return background_manager.submit_task(task, *args, inject_redis=True, **kwargs)
    except RuntimeError as e:
        logger.warning(f"Background task with Redis submission failed: {e}")
        return None

def is_background_manager_available() -> bool:
    return background_manager.is_initialized
