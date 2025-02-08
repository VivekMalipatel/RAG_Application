import logging
import asyncio
from aiobotocore.session import get_session
from app.config import settings

class MinIOClient:
    """Manages a single MinIO client session for the FastAPI app."""
    
    def __init__(self):
        self.session = get_session()
        self.client = None
        self.client_context = None

    async def connect(self):
        """Initialize an async MinIO client session with retries."""
        max_retries = 5
        for attempt in range(max_retries):
            try:
                if self.client is None:
                    self.client_context = self.session.create_client(
                        "s3",
                        endpoint_url=f"http://{settings.MINIO_ENDPOINT}",
                        aws_access_key_id=settings.MINIO_ACCESS_KEY,
                        aws_secret_access_key=settings.MINIO_SECRET_KEY,
                    )
                    self.client = await self.client_context.__aenter__()
                    buckets = await self.client.list_buckets()
                    logging.info(f"MinIO buckets: {buckets}")
                    logging.info("MinIO client initialized")
                return
            except Exception as e:
                logging.error(f"MinIO connection failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise e

    async def close(self):
        """Closes the MinIO client session."""
        if self.client_context:
            await self.client_context.__aexit__(None, None, None)
            self.client = None
            self.client_context = None
            logging.info("MinIO client closed")

minio_session = MinIOClient()

async def get_minio():
    """Yields the MinIO client for dependency injection."""
    await minio_session.connect()
    yield minio_session.client