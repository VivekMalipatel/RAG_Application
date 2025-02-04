import logging
import uuid
from app.core.storage_bin.minio.minio_handler import MinIOHandler
from app.core.storage_bin.postgres.postgres_handler import PostgresHandler
from app.core.kafka.kafka_handler import KafkaHandler

class RequestValidator:
    """
    Handles file upload validation, including type checks, name conflicts,
    and multipart upload approvals.
    """

    def __init__(self, minio_config: dict, db_url: str, kafka_config: dict):
        """
        Initializes request validator with MinIO, PostgreSQL, and Kafka handlers.

        Args:
            minio_config (dict): MinIO connection parameters (endpoint, access key, etc.).
            db_url (str): PostgreSQL database connection string.
            kafka_config (dict): Kafka topic details.
        """
        self.minio = MinIOHandler(**minio_config)
        self.db = PostgresHandler(db_url)
        self.kafka = KafkaHandler(**kafka_config)

    async def validate_request(self, request_data: dict, file_data: bytes = None):
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
            user_id = request_data["user_id"]
            file_name = request_data["file_name"]
            upload_id = request_data.get("upload_id")
            chunk_number = request_data.get("chunk_number")
            uploadapproval_id = request_data.get("uploadapproval_id")

            logging.info(f"Validating request for user: {user_id}, File: {file_name}")

            # Determine if it's a new file upload or an existing multipart chunk
            if not upload_id and not chunk_number and not uploadapproval_id:
                return await self._handle_new_file(request_data)
            else:
                return await self._handle_existing_upload(request_data, file_data)

        except Exception as e:
            logging.error(f"Unexpected error during request validation: {e}")
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
            total_chunks = request_data.get("total_chunks")

            # Step 1: Validate file type
            if not self._validate_file_type(mime_type):
                logging.error(f"File type {mime_type} not supported.")
                return {"success": False, "error": "Unsupported file type."}

            # Step 2: Check for duplicate file names in MinIO
            minio_path = f"{user_id}/{relative_path}/{file_name}"
            existing_file = await self.db.get_file_metadata(user_id, file_name)

            if existing_file:
                logging.error(f"File {file_name} already exists in {relative_path}.")
                return {"success": False, "error": "File name already exists."}

            # Step 3: Request a new multipart upload from MinIO
            upload_id = await self.minio.start_multipart_upload(minio_path)
            if not upload_id:
                logging.error("Failed to initiate multipart upload.")
                return {"success": False, "error": "Failed to start upload."}

            # Step 4: Generate upload approval ID and store request details in PostgreSQL
            uploadapproval_id = str(uuid.uuid4())

            # Store complete request details for tracking and approval verification later
            await self.db.insert_multipart_upload(
                user_id=user_id,
                file_name=file_name,
                upload_id=upload_id,
                uploadapproval_id=uploadapproval_id,
                relative_path=relative_path,
                file_size=file_size,
                mime_type=mime_type,
                total_chunks=total_chunks,
                uploaded_chunks=[],  # Use list instead of set (Postgres JSON compatibility)
            )

            logging.info(f"Upload approved for {file_name}, Upload ID: {upload_id}, Approval ID: {uploadapproval_id}")

            return {
                "success": True,
                "uploadapproval_id": uploadapproval_id,
                "upload_id": upload_id
            }

        except Exception as e:
            logging.error(f"Error during new file validation: {e}")
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

            # Step 1: Check if approval exists in PostgreSQL
            existing_upload = await self.db.get_multipart_upload(uploadapproval_id)
            if not existing_upload:
                logging.error(f"Invalid upload approval ID: {uploadapproval_id}")
                return {"success": False, "error": "Invalid upload request."}

            # Step 2: Verify metadata consistency
            uploaded_chunks = set(existing_upload["uploaded_chunks"])  # Convert to set for faster lookup
            if (
                existing_upload["file_name"] != file_name or
                existing_upload["upload_id"] != upload_id or
                existing_upload["file_size"] != file_size or
                existing_upload["user_id"] != user_id or
                existing_upload["total_chunks"] != total_chunks or
                existing_upload["relative_path"] != relative_path or
                existing_upload["mime_type"] != mime_type or
                chunk_number in uploaded_chunks
            ):
                logging.error("File metadata mismatch for upload request.")
                return {"success": False, "error": "File metadata mismatch."}

            # Step 3: If file data is provided, add the upload request to Kafka queue
            if file_data:
                kafka_payload = {
                    "user_id": user_id,
                    "file_name": file_name,
                    "relative_path": relative_path,
                    "upload_id": upload_id,
                    "chunk_number": chunk_number,
                    "file_data": file_data,
                    "file_size": file_size,
                    "mime_type": mime_type,
                    "total_chunks": total_chunks,
                    "uploadapproval_id": uploadapproval_id
                }
                await self.kafka.add_to_queue("file_upload_requests", kafka_payload)
                logging.info(f"Upload request added to Kafka: {file_name}, Chunk: {chunk_number}")

                return {"success": True, "message": "Upload request queued for processing."}
            
            logging.warning("No file data provided for multipart upload.")
            return {"success": False, "error": "No file data received."}

        except Exception as e:
            logging.error(f"Error handling existing upload request: {e}")
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