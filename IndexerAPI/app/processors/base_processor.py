from abc import ABC, abstractmethod
from typing import Dict, Any, List, Union, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class BaseProcessor(ABC):
    
    @abstractmethod
    async def process(self, data: Any, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        pass