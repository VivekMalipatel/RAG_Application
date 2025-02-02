import aiofiles
import os
import magic
import logging
from io import BytesIO
from app.core.storage_bin.minio_client import MinIOClient

class FileHandler:
    """Handles file saving, retrieval, and type detection asynchronously."""
    
    def __init__(self):
        self.minio = MinIOClient()
        self.file_mime = magic.Magic(mime=True)  # Used for MIME type detection

    async def detect_file_type(self, file_path: str) -> str:
        """Detects file MIME type asynchronously."""
        try:
            async with aiofiles.open(file_path, 'rb') as f:
                file_data = await f.read(2048)  # Read only the first few bytes for type detection
            return self.file_mime.from_buffer(file_data)
        except Exception as e:
            logging.error(f"Failed to detect file type for {file_path}: {e}")
            return "unknown"

    async def save_file(self, user_id: str, file_data: bytes, file_name: str) -> str:
        """
        Saves the file and uploads it to MinIO.

        - `user_id`: Owner of the file.
        - `file_data`: File content as bytes.
        - `file_name`: Name of the file.

        Returns:
            The MinIO file path if successful.
        """
        try:
            file_type = self.file_mime.from_buffer(file_data)  # Detect file type
            
            # Convert bytes to in-memory stream instead of writing to disk
            file_stream = BytesIO(file_data)

            # Upload to MinIO
            minio_path = await self.minio.upload_file(user_id, file_stream, file_name)
            logging.info(f"✅ File '{file_name}' uploaded to MinIO at {minio_path}")

            return minio_path
        except Exception as e:
            logging.error(f"Error saving file '{file_name}' for user {user_id}: {e}")
            return None

    async def retrieve_file(self, user_id: str, file_name: str) -> BytesIO:
        """
        Retrieves a file asynchronously from MinIO.

        Returns:
            File content as an in-memory stream.
        """
        try:
            file_stream = await self.minio.get_file(user_id, file_name)
            logging.info(f"✅ File '{file_name}' retrieved successfully.")
            return file_stream
        except Exception as e:
            logging.error(f"❌ Error retrieving file '{file_name}' for user {user_id}: {e}")
            return None