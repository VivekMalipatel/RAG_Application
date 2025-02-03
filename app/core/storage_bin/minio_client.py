import os
import logging
import asyncio
from io import BytesIO
from minio import Minio
from minio.error import S3Error
from dotenv import load_dotenv

load_dotenv()

class MinIOClient:
    """Handles MinIO file storage operations asynchronously."""

    def __init__(self):
        self.client = Minio(
            endpoint=os.getenv("MINIO_ENDPOINT", "localhost:9000"),
            access_key=os.getenv("MINIO_ROOT_USER"),
            secret_key=os.getenv("MINIO_ROOT_PASSWORD"),
            secure=False
        )
        self.bucket_prefix = os.getenv("MINIO_BUCKET_NAME", "omnirag-storage")

    def _get_bucket_name(self, user_id: str) -> str:
        """Returns the formatted bucket name for a user."""
        return f"{self.bucket_prefix}-{user_id}".lower()

    async def create_user_bucket(self, user_id: str) -> str:
        """Creates a MinIO bucket for a user if it does not exist."""
        bucket_name = self._get_bucket_name(user_id)
        try:
            if not self.client.bucket_exists(bucket_name):
                await asyncio.to_thread(self.client.make_bucket, bucket_name)
                logging.info(f"Created bucket: {bucket_name}")
        except S3Error as e:
            logging.error(f"MinIO Error while creating bucket '{bucket_name}': {e}")
            return None
        return bucket_name

    async def upload_file(self, user_id: str, file_data: bytes, file_name: str) -> str:
        """
        Uploads a file to MinIO asynchronously.

        Args:
            user_id (str): User ID for bucket association.
            file_data (bytes): File content in bytes.
            file_name (str): Name of the file.

        Returns:
            str: Full MinIO path of the uploaded file, or None on failure.
        """
        bucket_name = await self.create_user_bucket(user_id)
        if not bucket_name:
            return None

        object_name = f"{user_id}/{file_name}"

        try:
            file_stream = BytesIO(file_data)  # Use in-memory buffer instead of writing to disk
            await asyncio.to_thread(
                self.client.put_object, 
                bucket_name, object_name, file_stream, len(file_data)
            )
            logging.info(f"File '{file_name}' uploaded to MinIO at {bucket_name}/{object_name}")
            return f"{bucket_name}/{object_name}"
        except S3Error as e:
            logging.error(f"MinIO upload failed for '{file_name}': {e}")
            return None

    async def get_file(self, user_id: str, file_name: str) -> BytesIO:
        """
        Retrieves a file from MinIO asynchronously.

        Args:
            user_id (str): User ID for bucket association.
            file_name (str): Name of the file.

        Returns:
            BytesIO: File content in an in-memory stream, or None on failure.
        """
        bucket_name = self._get_bucket_name(user_id)
        object_name = f"{user_id}/{file_name}"

        try:
            file_stream = BytesIO()
            await asyncio.to_thread(
                self.client.fget_object, bucket_name, object_name, file_stream
            )
            logging.info(f"File '{file_name}' retrieved successfully from MinIO.")
            return file_stream
        except S3Error as e:
            logging.error(f"MinIO retrieval failed for '{file_name}': {e}")
            return None