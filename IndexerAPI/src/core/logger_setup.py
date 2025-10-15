import logging
from typing import Optional
from core.config import settings

def setup_logging(level_override: Optional[str] = None) -> logging.Logger:
    level_name = (level_override or settings.LOG_LEVEL or "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    return logging.getLogger("indexer")

logger = setup_logging()
