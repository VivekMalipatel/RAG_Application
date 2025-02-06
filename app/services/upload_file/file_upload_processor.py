import logging
from app.core.storage_bin.minio.minio_handler import MinIOHandler
from app.core.storage_bin.postgres.postgres_handler import PostgresHandler
from app.core.cache.redis_cache import RedisCache

class FileUploadProcessor:
    """
    Processes file upload requests, uploads to MinIO, updates PostgreSQL,
    and handles multipart upload completion.
    """

    def __init__(self, minio: MinIOHandler , db: PostgresHandler, cache: RedisCache):
        """
        Initializes Kafka handler, MinIO client, and PostgreSQL handler.

        Args:
            kafka_config (dict): Kafka connection details.
            minio_config (dict): MinIO connection details.
            db_url (str): PostgreSQL database connection URL.
        """
        self.minio = minio
        self.db = db
        self.cache = cache

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
            file_data = upload_request["file_data"]
            file_size = upload_request["file_size"]
            mime_type = upload_request["mime_type"]

            minio_path = f"{user_id}/{relative_path}/{file_name}"
            #minio_path = f"{user_id}/{relative_path}"

            logging.info(f"Processing chunk {chunk_number}/{total_chunks} for {file_name}")
            print(f"Processing chunk {chunk_number}/{total_chunks} for {file_name}")

            logging.debug(f"Uploading chunk {chunk_number} to MinIO at path {minio_path}")
            print(f"Uploading chunk {chunk_number} to MinIO at path {minio_path}")

            print("Upload the chunk to MinIO")
            part_info = await self.minio.upload_part(minio_path, upload_id, chunk_number, file_data)

            if not part_info:
                logging.error("Failed to upload chunk. Retrying...")
                print("Failed to upload chunk. Retrying...")
            
            print("Upload the chunk to MinIO is done")

            logging.debug("Checking if all chunks are uploaded...")
            response = await self.check_and_finalize_upload(upload_id, total_chunks, relative_path)
            if response:
                await self.update_postgres(response["minio_path"], user_id, file_name, file_size, response["etag"], mime_type)
                # Delete the approval ID from Redis
                await self.cache.delete(upload_request["uploadapproval_id"])
                logging.info(f"Upload completed for {file_name}.")

        except Exception as e:
            logging.error(f"Error processing upload : {e}")
            raise (e)

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
            uploaded_chunks = await self.minio.get_uploaded_parts(relative_path,upload_id)
            if len(uploaded_chunks) == total_chunks:
                
                logging.debug("Finalizing upload in MinIO...")
                # Call MinIO to complete the multipart upload
                minio_response = await self.minio.complete_multipart_upload(
                    relative_path, upload_id
                )

                if minio_response:
                    logging.info("Upload finalized.")
                    return minio_response
            self.db.close() # Close the database connection

        except Exception as e:
            logging.error(f"Error finalizing upload: {e}")

        return

    async def update_postgres(self, minio_path: str, user_id: str, file_name: str, file_size: int, etag: str, mime_type: str):
        """
        Updates the PostgreSQL `files` table with the uploaded file metadata.

        """
        try:

            logging.debug("Inserting file metadata into PostgreSQL...")
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
            raise (e)