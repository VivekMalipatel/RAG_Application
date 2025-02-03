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

    async def detect_file_type(self, file_data: bytes) -> str:
        """
        Detects file MIME type from bytes.

        Args:
            file_data (bytes): The file content.

        Returns:
            str: Detected MIME type (e.g., 'application/pdf', 'image/png').
        """
        try:
            return self.file_mime.from_buffer(file_data)
        except Exception as e:
            logging.error(f"Failed to detect file type: {e}")
            return "application/octet-stream"  # Fallback type

    async def save_file(self, user_id: str, file_data: bytes, file_name: str) -> str:
        """
        Saves the file and uploads it to MinIO.

        Args:
            user_id (str): Owner of the file.
            file_data (bytes): File content as bytes.
            file_name (str): Name of the file.

        Returns:
            dict: Success or error message with MinIO file path.
        """
        try:
            # Detect file type
            file_type = await self.detect_file_type(file_data)
            logging.info(f"Detected file type for '{file_name}': {file_type}")

            # Convert bytes to in-memory stream for MinIO
            file_stream = BytesIO(file_data)

            # Upload to MinIO
            minio_path = await self.minio.upload_file(user_id, file_stream, file_name)
            logging.info(f"File '{file_name}' uploaded successfully to MinIO at {minio_path}")

            return {"success": True, "file_path": minio_path}
        except Exception as e:
            logging.error(f"Error saving file '{file_name}' for user {user_id}: {e}")
            return {"success": False, "error": str(e)}

    async def retrieve_file(self, user_id: str, file_name: str) -> BytesIO:
        """
        Retrieves a file asynchronously from MinIO.

        Args:
            user_id (str): Owner of the file.
            file_name (str): Name of the file to retrieve.

        Returns:
            BytesIO: File content as an in-memory stream, or None if an error occurs.
        """
        try:
            async with self.minio.get_file(user_id, file_name) as file_stream:
                logging.info(f"File '{file_name}' retrieved successfully.")
                return file_stream
        except Exception as e:
            logging.error(f"Error retrieving file '{file_name}' for user {user_id}: {e}")
            return None