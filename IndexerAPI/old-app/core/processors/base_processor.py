from abc import ABC, abstractmethod
from typing import Dict, Any, List, Union, Optional
import logging

logger = logging.getLogger(__name__)

class BaseProcessor(ABC):
    
    @abstractmethod
    async def process(self, task_message) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement process method")