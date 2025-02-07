import logging
import uuid
import asyncio
from app.core.storage_bin.minio.minio_handler import MinIOHandler
from app.core.storage_bin.postgres.postgres_handler import PostgresHandler
from app.services.upload_file.file_upload_processor import FileUploadProcessor
from app.core.cache.redis_cache import RedisCache

class RequestValidator:
    """
    Handles file upload validation, including type checks, name conflicts,
    and multipart upload approvals.
    """

    def __init__(self, minio_config: dict, db_url: str, redis_url: str):
        """
        Initializes request validator with MinIO and PostgreSQL handlers.

        Args:
            minio_config (dict): MinIO connection parameters.
            db_url (str): PostgreSQL database connection string.
        """
        self.minio = MinIOHandler(**minio_config)
        self.db = PostgresHandler(db_url)
        self.cache = RedisCache(redis_url)

        self.approval_cache = {}
        self.file_upload_processor = FileUploadProcessor(self.minio, self.db, self.cache)
        

    async def validate_request(self, request_data: dict, file_data: bytes = None):
        
        # Connect to database and Redis cache
        try:
            await self.db.init_db()
            await self.cache.connect()
        except Exception as e:
            logging.error(f"Error initializing database/redis cache: {e}")
            return {"success": False, "error": "Internal server error."}
        
        """
        Validates the incoming file upload request.

        Args:
            request_data (dict): Incoming file details.
                - user_id (str)
                - file_name (str)
                - relative_path (str)
                - file_size (int)
                - mime_type (str)
                - upload_id (str, optional)
                - total_chunks (int)
                - chunk_number (int, optional)
                - uploadapproval_id (str, optional)
            file_data (bytes, optional): Binary file data if provided.

        Returns:
            dict: Response indicating approval or rejection.
        """
        try:
            logging.info(f"Validating request for user: {request_data['user_id']}, File: {request_data['file_name']}")

            # Determine if it's a new file upload or an existing multipart chunk
            if not request_data.get("upload_id") and not request_data.get("chunk_number") and not request_data.get("uploadapproval_id") and (file_data is None):
                logging.info(f"New file upload request for {request_data['file_name']}") #used for testing
                return await self._handle_new_file(request_data)
            return await self._handle_existing_upload(request_data, file_data)

        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            return {"success": False, "error": "Internal server error."}

    async def _handle_new_file(self, request_data: dict):
        """
        Handles validation for a new file upload.

        Args:
            request_data (dict): Incoming file details.

        Returns:
            dict: Approval or rejection response.
        """
        try:
            user_id = request_data["user_id"]
            file_name = request_data["file_name"]
            relative_path = request_data["relative_path"]
            file_size = request_data["file_size"]
            mime_type = request_data["mime_type"]
            total_chunks = request_data["total_chunks"]

            # Validate file type
            if not self._validate_file_type(mime_type):
                logging.error("Unsupported file type.")
                return {"success": False, "error": "Unsupported file type."}

            minio_path = f"{user_id}/{relative_path}/{file_name}"

            # Check for duplicate file names in PostgreSQL
            existing_file = await self.db.get_file_metadata(user_id, file_name)
            if existing_file:
                logging.error(f"File '{file_name}' already exists in {relative_path}.")
                return {"success": False, "error": "File name already exists."}

            # Request a new multipart upload from MinIO
            upload_id = await self.minio.start_multipart_upload(minio_path)
            if not upload_id:
                logging.error(f"Failed to initiate multipart upload for {file_name}")
                return {"success": False, "error": "Failed to start upload."}

            # Generate upload approval ID and store request details in Redis
            uploadapproval_id = str(uuid.uuid4())

            # Store the approval ID and upload ID combination in Redis
            redis_value = {
                "upload_id": upload_id,
                "relative_path": relative_path,
                "total_chunks": total_chunks
            }

            await self.cache.set(uploadapproval_id, redis_value)

            logging.info(f"Upload approved for {file_name}, Upload ID: {upload_id}, Approval ID: {uploadapproval_id}")

            return {
                "success": True,
                "uploadapproval_id": uploadapproval_id,
                "upload_id": upload_id
            }

        except Exception as e:
            logging.error(f"Error validating new file: {e}")
            return {"success": False, "error": "Internal server error."}

    async def _handle_existing_upload(self, request_data: dict, file_data: bytes = None):

        """
        Handles validation for an ongoing multipart upload request.

        Args:
            request_data (dict): Incoming file details.
            file_data (bytes, optional): The binary file data.

        Returns:
            dict: Approval or rejection response.
        """
        try:
            user_id = request_data["user_id"]
            file_name = request_data["file_name"]
            uploadapproval_id = request_data["uploadapproval_id"]
            upload_id = request_data["upload_id"]
            chunk_number = request_data["chunk_number"]
            total_chunks = request_data["total_chunks"]
            relative_path = request_data["relative_path"]
            file_size = request_data["file_size"]
            mime_type = request_data["mime_type"]

            # Check if approval exists in local or Redis cache
            if uploadapproval_id in self.approval_cache:
                approval_data = self.approval_cache[uploadapproval_id]
            else:
                approval_data = await self.cache.get(uploadapproval_id)
                if not approval_data:
                    logging.error(f"Upload approval ID {uploadapproval_id} not found.")
                    return {"success": False, "error": "Invalid upload approval ID."}
                self.approval_cache[uploadapproval_id] = approval_data  # Cache it
        
            # Verify metadata consistency
            metadata_mismatch = (
                approval_data["relative_path"] != relative_path or
                approval_data["total_chunks"] != total_chunks
            )
            print("metadata_mismatch=", metadata_mismatch)

            if metadata_mismatch:
                logging.error("File metadata mismatch.")
                return {"success": False, "error": "Metadata mismatch."}
            
            print("file_data=", file_data)

            if file_data:
                upload_payload = {
                    "user_id": user_id,
                    "file_name": file_name,
                    "relative_path": relative_path,
                    "upload_id": upload_id,
                    "chunk_number": chunk_number,
                    "file_size": file_size,
                    "mime_type": mime_type,
                    "total_chunks": total_chunks,
                    "uploadapproval_id": uploadapproval_id,
                    "file_data": file_data
                }

                # Run upload in the background to avoid blocking request processing
                asyncio.create_task(self.file_upload_processor.process_upload(upload_payload))
                print(f"Chunk upload started for {file_name}, chunk {chunk_number}.")

                return {"success": True, "message": "Chunk upload started."}
            
            logging.warning(f"No file data received for {file_name}, chunk {chunk_number}.")
            return {"success": False, "error": "No file data received."}

        except Exception as e:
            logging.error(f"Error handling existing upload: {e}")
            return {"success": False, "error": "Internal server error."}

    def _validate_file_type(self, mime_type: str) -> bool:
        """
        Validates if the uploaded file type is allowed.

        Args:
            mime_type (str): MIME type of the uploaded file.

        Returns:
            bool: True if allowed, False otherwise.
        """
        allowed_types = ["image/png", "image/jpeg", "application/pdf", "text/plain"]
        return mime_type in allowed_types

    def clear_approval(self, uploadapproval_id: str):
        if uploadapproval_id in self.approval_cache:
            del self.approval_cache[uploadapproval_id]
        # Optionally also remove from Redis
        asyncio.create_task(self.cache.delete(uploadapproval_id))