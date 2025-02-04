import logging
from app.core.kafka.kafka_handler import KafkaHandler
from app.core.storage_bin.minio.minio_handler import MinIOHandler
from app.core.storage_bin.postgres.postgres_handler import PostgresHandler
from app.core.cache.redis_cache import RedisCache
import hashlib

class FileUploadProcessor:
    """
    Consumes file upload requests from Kafka, validates, uploads to MinIO,
    updates PostgreSQL, and handles multipart upload completion.
    """

    def __init__(self, kafka_config: dict, minio_config: dict, db_url: str, redis_url: str):
        """
        Initializes Kafka handler, MinIO client, and PostgreSQL handler.

        Args:
            kafka_config (dict): Kafka connection details.
            minio_config (dict): MinIO connection details.
            db_url (str): PostgreSQL database connection URL.
        """
        self.kafka = KafkaHandler(**kafka_config)
        self.minio = MinIOHandler(**minio_config)
        self.db = PostgresHandler(db_url)
        self.redis = RedisCache(redis_url)

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
            chunk_storage_key = upload_request["chunk_storage_key"]
            stored_chunk_hash = upload_request["chunk_hash"]
            retries = upload_request.get("retries", 0)

            logging.info(f"Processing chunk {chunk_number}/{total_chunks} for {file_name}")

            # Step 1: Validate Upload Approval in PostgreSQL
            existing_upload = await self.db.get_multipart_upload(uploadapproval_id)
            if not existing_upload:
                logging.error(f"Invalid upload approval ID: {uploadapproval_id}")
                return
            
            file_data = await self.redis.get(chunk_storage_key)
            if not file_data:
                logging.error(f"Chunk {chunk_number} not found in Redis for {file_name}. Retrying...")
                await self.retry_upload(upload_request, retries)
                return
            
            computed_hash = hashlib.md5(file_data).hexdigest()
            if computed_hash != stored_chunk_hash:
                logging.error(f"Chunk {chunk_number} hash mismatch! Upload rejected.")
                return  # Reject the upload if corruption detecteds

            # Step 2: Ensure chunk has not already been uploaded
            if chunk_number in existing_upload["uploaded_chunks"]:
                logging.warning(f"Chunk {chunk_number} already uploaded for {file_name}. Skipping...")
                return

            # Step 3: Upload Chunk to MinIO
            minio_path = f"{user_id}/{relative_path}/{file_name}"
            part_info = await self.minio.upload_part(minio_path, upload_id, chunk_number, file_data)

            if not part_info:
                logging.error(f"Failed to upload chunk {chunk_number} for {file_name}. Retrying...")
                await self.retry_upload(upload_request, retries)
                return

            # Step 4: Update PostgreSQL with chunk details
            etag = part_info["etag"]
            await self.db.update_multipart_part(upload_id, chunk_number, etag)
            await self.redis.delete(chunk_storage_key)
            await self.redis.delete(stored_chunk_hash)

            # Step 5: Check if Upload is Complete
            is_complete = await self.check_and_finalize_upload(upload_id, total_chunks)
            if is_complete:
                logging.info(f"Upload completed for {file_name}.")

        except Exception as e:
            logging.error(f"Error processing upload request: {e}")
            await self.retry_upload(upload_request, retries)

    async def check_and_finalize_upload(self, upload_id: str, total_chunks: int):
        """
        Checks if all chunks are uploaded and finalizes the upload.

        Args:
            upload_id (str): Multipart upload ID.
            total_chunks (int): Total number of chunks.

        Returns:
            bool: True if upload is finalized, else False.
        """
        try:
            upload_status = await self.db.get_multipart_upload(upload_id)
            uploaded_chunks = upload_status["uploaded_chunks"]

            if len(uploaded_chunks) == total_chunks:
                # Finalize the upload in MinIO
                all_parts = [{"part_number": k, "etag": v} for k, v in uploaded_chunks.items()]
                success = await self.minio.complete_multipart_upload(upload_status["relative_path"], upload_id, all_parts)

                if success:
                    await self.db.complete_multipart_upload(upload_id)
                    logging.info(f"Upload {upload_id} finalized in MinIO.")
                    return True

        except Exception as e:
            logging.error(f"Error finalizing upload: {e}")

        return False

    async def retry_upload(self, upload_request: dict):
        """
        Pushes failed chunk uploads to the Kafka delayed queue.

        Args:
            upload_request (dict): The original upload request.
        """
        try:
            await self.kafka.add_to_queue("file_upload_failures_delayed", upload_request)
            logging.error(f"Failed to upload chunk {upload_request['chunk_number']}. Sent to delayed queue.")

        except Exception as e:
            logging.error(f"Error in retry mechanism: {e}")