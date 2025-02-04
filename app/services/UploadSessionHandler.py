import logging
import uuid
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmakers
from sqlalchemy import select
from app.core.storage_bin.postgres.postgres_handler import PostgresHandler
from app.core.vectorstore.qdrant_client import QdrantDB

class UploadSessionHandler:
    """Manages upload session creation and validation before file data is sent."""

    def __init__(self, db_session: AsyncSession, postgres: PostgresHandler):
        self.db_session = db_session
        self.postgres = postgres

    async def create_or_resume_upload(self, user_id: str, file_name: str, file_size: int, relative_path: str, upload_id: str = None):
        """
        Handles the request for a new or resumed multipart upload.

        Args:
            user_id (str): User ID.
            file_name (str): Name of the file.
            file_size (int): Total file size.
            relative_path (str): Path in user's storage bin.
            upload_id (str, optional): Existing upload ID (for resuming uploads). Defaults to None.

        Returns:
            dict: Upload session response.
        """
        try:
            # 1️⃣ Check if file name is already used in the folder
            existing_file = await self.postgres.get_file_metadata(user_id, file_name)
            if existing_file and not upload_id:
                return {"status": "error", "message": "File name already exists in this folder."}

            # 2️⃣ If an upload_id is provided, check which chunks are missing
            if upload_id:
                multipart_data = await self.postgres.get_multipart_upload(upload_id)
                if multipart_data:
                    missing_chunks = [part for part in range(1, multipart_data.total_parts + 1) if part not in multipart_data.uploaded_parts]
                    return {"status": "resume", "upload_id": upload_id, "missing_chunks": missing_chunks}

            # 3️⃣ Generate a new upload ID for fresh uploads
            new_upload_id = str(uuid.uuid4())
            await self.postgres.insert_multipart_upload(user_id, file_name, total_parts=0, upload_id=new_upload_id)

            return {"status": "new", "upload_id": new_upload_id, "message": "Upload session started."}

        except Exception as e:
            logging.error(f"Error in upload session handling: {e}")
            return {"status": "error", "message": "Internal server error."}