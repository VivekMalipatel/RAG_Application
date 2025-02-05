import logging
from app.core.kafka.kafka_handler import KafkaHandler
from app.core.storage_bin.minio.minio_handler import MinIOHandler
from app.core.storage_bin.postgres.postgres_handler import PostgresHandler
from app.core.cache.redis_cache import RedisCache
import hashlib

class FileUploadProcessor:
    """
    Processes file upload requests, uploads to MinIO, updates PostgreSQL,
    and handles multipart upload completion.
    """

    def __init__(self, minio_config: dict, db_url: str):
        """
        Initializes Kafka handler, MinIO client, and PostgreSQL handler.

        Args:
            kafka_config (dict): Kafka connection details.
            minio_config (dict): MinIO connection details.
            db_url (str): PostgreSQL database connection URL.
        """
        self.minio = MinIOHandler(**minio_config)
        self.db = PostgresHandler(db_url)

    async def process_upload(self, upload_request: dict):
        """
        Processes a file upload request.

        Args:
            upload_request (dict): File chunk details received from Kafka.
        """
        try:

            user_id = upload_request["user_id"]
            file_name = upload_request["file_name"]
            upload_id = upload_request["upload_id"]
            chunk_number = upload_request["chunk_number"]
            total_chunks = upload_request["total_chunks"]
            relative_path = upload_request["relative_path"]
            uploadapproval_id = upload_request["uploadapproval_id"]
            file_data = upload_request["file_data"]
            file_size = upload_request["file_size"]
            mime_type = upload_request["mime_type"]

            logging.info(f"Processing chunk {chunk_number}/{total_chunks} for {file_name}")

            # **Step 5: Upload Chunk to MinIO**
            minio_path = f"{user_id}/{relative_path}/{file_name}"
            part_info = await self.minio.upload_part(minio_path, upload_id, chunk_number, file_data)
            # part_info : {"part_number": chunk_number, "etag": response.etag}

            if not part_info:
                logging.error("Failed to upload chunk. Retrying...")

            # Step 7: Check if Upload is Complete
            response = await self.check_and_finalize_upload(upload_id, total_chunks)
            if response:
                await self.update_postgres(response["minio_path"], user_id, file_name, file_size, response["etag"], mime_type)
                logging.info(f"Upload completed for {file_name}.")

        except Exception as e:
            logging.error(f"Error processing upload: {e}")

    async def check_and_finalize_upload(self, upload_id: str, total_chunks: int, relative_path: str):
        """
        Checks if all chunks are uploaded and finalizes the upload.

        Args:
            upload_id (str): Multipart upload ID.
            total_chunks (int): Total number of chunks.

        Returns:
            bool: True if upload is finalized, else False.
        """
        try:
            uploaded_chunks = await self.minio.get_uploaded_parts(upload_id)
            if len(uploaded_chunks) == total_chunks:
                
                # Call MinIO to complete the multipart upload
                minio_response = await self.minio.complete_multipart_upload(
                    relative_path, upload_id
                )

                if minio_response:
                    logging.info("Upload finalized.")
                    return minio_response

        except Exception as e:
            logging.error(f"Error finalizing upload: {e}")

        return

    async def update_postgres(self, minio_path: str, user_id: str, file_name: str, file_size: int, etag: str, mime_type: str):
        """
        Updates the PostgreSQL `files` table with the uploaded file metadata.

        Args:
            success_message (dict): File metadata from Kafka.
        """
        try:

            # Insert into files table with MinIO metadata
            await self.db.insert_file_metadata(
                user_id=user_id,
                file_name=file_name,
                file_path=minio_path,  # Store MinIO path directly
                file_size=file_size,
                file_hash=etag,  # Use MinIO ETag for integrity verification
                mime_type=mime_type,
            )

            logging.info("Successfully updated PostgreSQL.")

        except Exception as e:
            logging.error(f"Failed to update PostgreSQL: {e}")