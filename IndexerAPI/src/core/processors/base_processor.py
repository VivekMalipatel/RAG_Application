from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseProcessor(ABC):
    @abstractmethod
    async def process(self, task_message) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement process method")
