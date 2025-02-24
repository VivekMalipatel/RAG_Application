import logging
import asyncio
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import Distance, VectorParams
from app.config import settings

class QdrantSession:
    """
    Manages a global Qdrant client session for async operations.
    """

    def __init__(self):
        self.client = None
        self.lock = asyncio.Lock()

    async def connect(self):
        """Initialize Qdrant client."""
        async with self.lock:
            if not self.client:
                try:
                    self.client = AsyncQdrantClient(url=settings.QDRANT_URL)
                    logging.info("Connected to Qdrant successfully.")
                except Exception as e:
                    logging.error(f"Failed to initialize Qdrant session: {str(e)}")
                    raise

    async def close(self):
        """Close Qdrant session if needed."""
        async with self.lock:
            if self.client:
                self.client = None  # Qdrant client does not require explicit closure
                logging.info("Qdrant session closed.")

# Instantiate a global Qdrant session
qdrant_session = QdrantSession()