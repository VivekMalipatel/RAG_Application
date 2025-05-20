import threading
import logging
import concurrent.futures
from typing import Callable, Any, Dict
import os
import time

logger = logging.getLogger(__name__)

class CPUThreadPool:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(CPUThreadPool, cls).__new__(cls)
                total_cpus = os.cpu_count() or 4
                reserved_cpus = 1
                thread_multiplier = 1.5  
                max_workers = max(1, int((total_cpus - reserved_cpus) * thread_multiplier))
                
                cls._instance._executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=max_workers,
                    thread_name_prefix="cpu_worker"
                )
                cls._instance._stats = {
                    "tasks_submitted": 0,
                    "tasks_completed": 0,
                    "tasks_failed": 0,
                    "avg_execution_time": 0,
                }
                logger.info(f"CPU Thread Pool initialized with {max_workers} workers")
            return cls._instance
    
    def submit(self, fn: Callable, *args, **kwargs) -> concurrent.futures.Future:
        self._stats["tasks_submitted"] += 1
        start_time = time.time()
        
        def wrapped_fn(*args, **kwargs):
            try:
                result = fn(*args, **kwargs)
                self._stats["tasks_completed"] += 1
                execution_time = time.time() - start_time
                
                prev_avg = self._stats["avg_execution_time"]
                prev_count = self._stats["tasks_completed"]
                self._stats["avg_execution_time"] = (prev_avg * (prev_count - 1) + execution_time) / prev_count
                
                return result
            except Exception as e:
                self._stats["tasks_failed"] += 1
                logger.error(f"Error in thread pool task: {str(e)}")
                raise
        
        return self._executor.submit(wrapped_fn, *args, **kwargs)
    
    def map(self, fn: Callable, *iterables, timeout=None, chunksize=1):
        return self._executor.map(fn, *iterables, timeout=timeout, chunksize=chunksize)
    
    def shutdown(self, wait=True):
        return self._executor.shutdown(wait=wait)
    
    def get_stats(self) -> Dict[str, Any]:
        stats = self._stats.copy()
        stats["active_threads"] = len([t for t in threading.enumerate() 
                                     if t.name.startswith("cpu_worker")])
        return stats
    
    @property
    def max_workers(self) -> int:
        return self._executor._max_workers


def get_cpu_thread_pool() -> CPUThreadPool:
    return CPUThreadPool()
