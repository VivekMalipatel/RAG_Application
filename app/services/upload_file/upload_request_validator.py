import logging
import uuid
import asyncio
from app.core.storage_bin.minio import MinIOHandler
from app.core.db_handler import DocumentHandler
from app.services.upload_file import FileUploadProcessor
from app.core.cache import RedisCache
from app.config import settings

class RequestValidator:
    """
    Handles file upload validation, including type checks, name conflicts,
    and multipart upload approvals.
    """

    def __init__(self):

        self.minio = MinIOHandler()
        self.db = DocumentHandler()
        self.cache = RedisCache()

        self.approval_cache = {}
        self.file_upload_processor = FileUploadProcessor(self.approval_cache)
        

    async def validate_request(self, request_data: dict, file_data: bytes = None):
        
        try:
            logging.info(f"Validating request for user: {request_data['user_id']}, File: {request_data['file_name']}")

            # Determine if it's a new file upload or an existing multipart chunk
            if not request_data.get("upload_id") and not request_data.get("chunk_number") and not request_data.get("approval_id") and (file_data is None):
                return await self._handle_new_file(request_data)
            return await self._handle_existing_upload(request_data, file_data)

        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            return {"success": False, "error": "Internal server error."}

    async def _handle_new_file(self, request_data: dict):

        try:
            user_id = request_data["user_id"]
            file_name = request_data["file_name"]
            local_file_path = request_data["local_file_path"]
            relative_path = request_data["relative_path"]
            mime_type = request_data["mime_type"]
            file_size = request_data["file_size"]
            total_chunks = request_data["total_chunks"]

            # Validate file type
            if not self._validate_file_type(mime_type):
                logging.info("Unsupported file type.")
                return {"success": False, "error": "Unsupported file type."}

            minio_path = f"{user_id}/{relative_path}/{file_name}"

            # Check for duplicate file names in PostgreSQL
            existing_file = await self.db.get_document_metadata(user_id, minio_path)
            if existing_file:
                logging.info(f"File '{file_name}' already exists in {relative_path}.")
                return {"success": False, "error": "File name already exists."}

            # Request a new multipart upload from MinIO
            upload_id = await self.minio.start_multipart_upload(minio_path)
            if not upload_id:
                logging.error(f"Failed to initiate multipart upload for {file_name}")
                return {"success": False, "error": "Failed to start upload."}

            # Generate upload approval ID and store request details in Redis
            approval_id = str(uuid.uuid4())

            # Store the approval ID and upload ID combination in Redis
            redis_value = {
                "user_id": user_id,
                "file_name": file_name,
                "local_file_path": local_file_path,
                "relative_path": relative_path,
                "mime_type": mime_type,
                "file_size": file_size,
                "total_chunks": total_chunks,
                "upload_id": upload_id
            }

            await self.cache.set(approval_id, redis_value)

            logging.info(f"Upload approved for {file_name}, Upload ID: {upload_id}, Approval ID: {approval_id}")

            return {
                "success": True,
                "approval_id": approval_id,
                "upload_id": upload_id,
                "message": "Upload approved."
            }

        except Exception as e:
            logging.error(f"Error validating new file: {e}")
            return {"success": False, "error": "Internal server error."}

    async def _handle_existing_upload(self, request_data: dict, file_data: bytes = None):

        try:

            user_id = request_data["user_id"]
            file_name = request_data["file_name"]
            local_file_path = request_data["local_file_path"]
            relative_path = request_data["relative_path"]
            mime_type = request_data["mime_type"]
            file_size = request_data["file_size"]
            total_chunks = request_data["total_chunks"]
            approval_id = request_data["approval_id"]
            upload_id = request_data["upload_id"]
            chunk_number = request_data["chunk_number"]
        
            if approval_id in self.approval_cache:
                approval_data = self.approval_cache[approval_id]
            else:
                approval_data = await self.cache.get(approval_id)
                if not approval_data:
                    logging.info(f"Upload approval ID {approval_id} not found.")
                    return {"success": False, "error": "You do not have permission to upload this file."}
                self.approval_cache[approval_id] = approval_data
        
            # Verify metadata consistency
            metadata_mismatch = (
                user_id != approval_data["user_id"] or
                file_name != approval_data["file_name"] or
                local_file_path != approval_data["local_file_path"] or
                relative_path != approval_data["relative_path"] or
                mime_type != approval_data["mime_type"] or
                file_size != approval_data["file_size"] or
                total_chunks != approval_data["total_chunks"] or
                upload_id != approval_data["upload_id"]
            )

            if metadata_mismatch:
                logging.info("File metadata mismatch.")
                return {"success": False, "error": "Please re-upload the correct file."}

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
                    "approval_id": approval_id,
                    "file_data": file_data
                }

                asyncio.create_task(self.file_upload_processor.process_upload(upload_payload))

                return {"success": True, "message": "Chunk upload started."}
            
            logging.error(f"No file data received for {file_name}, chunk {chunk_number}.")
            return {"success": False, "error": "No file data received."}

        except Exception as e:
            logging.error(f"Error handling existing upload: {e}")
            return {"success": False, "error": "Internal server error."}

    def _validate_file_type(self, mime_type: str) -> bool:

        allowed_types = ["image/png", "image/jpeg", "application/pdf", "text/plain"]
        return mime_type in allowed_types

    def clear_approval(self, approval_id: str):
        if approval_id in self.approval_cache:
            del self.approval_cache[approval_id]
        # Optionally also remove from Redis
        asyncio.create_task(self.cache.delete(approval_id))