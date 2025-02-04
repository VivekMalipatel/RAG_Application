import logging
import hashlib
from typing import Dict, Optional
from app.core.storage_bin.postgres.postgres_handler import PostgresHandler

class FileValidator:
    """
    Handles file validation before sending an upload request to Kafka.
    """

    def __init__(self, db_handler: PostgresHandler, allowed_extensions: Optional[list] = None):
        """
        Initializes the File Validator.

        Args:
            db_handler (PostgresHandler): PostgreSQL handler instance for metadata checking.
            allowed_extensions (list, optional): List of allowed file extensions. Defaults to common file types.
        """
        self.db_handler = db_handler
        self.allowed_extensions = allowed_extensions or ["pdf", "jpg", "png", "txt", "mp4", "mp3"]

    async def validate_upload(self, user_id: str, file_name: str, file_data: bytes, upload_id: Optional[str] = None) -> Dict:
        """
        Validates the upload request and determines its category.

        Args:
            user_id (str): The ID of the user uploading the file.
            file_name (str): The name of the file being uploaded.
            file_data (bytes): The file data in bytes.
            upload_id (Optional[str]): If provided, this indicates a multipart upload chunk.

        Returns:
            dict: Validation response including status and required metadata.
        """
        file_extension = file_name.split(".")[-1].lower()
        if file_extension not in self.allowed_extensions:
            return {"status": "error", "message": f"File type '{file_extension}' not allowed."}

        # Calculate file hash (SHA256) to check for duplicates
        file_hash = self.calculate_hash(file_data)
        
        if upload_id:
            # Case 1: Multipart Upload Chunk Handling
            return await self._handle_multipart_chunk(user_id, upload_id, file_name, file_hash)
        else:
            # Case 2 & 3: New Upload (Check for duplicates)
            return await self._handle_new_upload(user_id, file_name, file_hash, len(file_data))

    def calculate_hash(self, file_data: bytes) -> str:
        """
        Calculates the SHA-256 hash of the file content.

        Args:
            file_data (bytes): The file content.

        Returns:
            str: The computed hash value.
        """
        return hashlib.sha256(file_data).hexdigest()

    async def _handle_multipart_chunk(self, user_id: str, upload_id: str, file_name: str, file_hash: str) -> Dict:
        """
        Handles a multipart upload chunk by ensuring it belongs to an active multipart session.

        Args:
            user_id (str): The user ID.
            upload_id (str): The multipart upload session ID.
            file_name (str): The file name.
            file_hash (str): The file hash.

        Returns:
            dict: Status response for chunk validation.
        """
        upload_session = await self.db_handler.get_multipart_upload(upload_id)

        if not upload_session:
            return {"status": "error", "message": f"Multipart upload session '{upload_id}' not found."}

        if upload_session.user_id != user_id:
            return {"status": "error", "message": "Permission denied for multipart upload."}

        return {"status": "valid", "upload_id": upload_id, "message": "Chunk accepted."}

    async def _handle_new_upload(self, user_id: str, file_name: str, file_hash: str, file_size: int) -> Dict:
        """
        Handles a new file upload by checking naming conflicts and duplicate hashes.

        Args:
            user_id (str): The user ID.
            file_name (str): The name of the file.
            file_hash (str): The computed hash of the file.
            file_size (int): The size of the file in bytes.

        Returns:
            dict: Validation status and metadata.
        """
        existing_file = await self.db_handler.get_file_by_name(user_id, file_name)

        if existing_file:
            return {"status": "error", "message": "File with the same name already exists. Rename or overwrite."}

        existing_hash = await self.db_handler.get_file_by_hash(file_hash)

        if existing_hash:
            # Case 3: The file already exists, just link it to the user
            new_file_id = await self.db_handler.insert_file_reference(user_id, existing_hash.file_id, file_name)
            return {
                "status": "duplicate",
                "file_id": new_file_id,
                "message": "File already exists. User reference updated."
            }

        return {
            "status": "valid",
            "file_name": file_name,
            "file_hash": file_hash,
            "file_size": file_size,
            "message": "New file validated. Proceed with upload."
        }