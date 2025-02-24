import logging
from app.core.storage_bin.minio.minio_handler import MinIOHandler
from app.core.db_handler import DocumentHandler
from app.core.cache.redis_cache import RedisCache

class FileUploadProcessor:
    """
    Processes file upload requests, uploads to MinIO, updates PostgreSQL,
    and handles multipart upload completion.
    """

    def __init__(self, approval_cache: dict):

        self.minio = MinIOHandler()
        self.db = DocumentHandler()
        self.cache = RedisCache()
        self.approval_cache = approval_cache

    async def process_upload(self, upload_request: dict):

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
            approval_id = upload_request.get("approval_id")

            minio_path = f"{user_id}/{relative_path}/{file_name}"

            logging.info(f"Processing chunk {chunk_number}/{total_chunks} for {minio_path}...")

            part_info = await self.minio.upload_part(minio_path, upload_id, chunk_number, file_data)
            if not part_info:
                logging.error("Failed to upload chunk. Retrying...")

            response = await self.check_and_finalize_upload(upload_id, total_chunks, minio_path)
            if response:
                await self.update_postgres(f"{response["Bucket"]}/{response["Key"]}", user_id, file_name, file_size, response["ETag"], mime_type)
                await self.cache.delete(upload_request["approval_id"])
                if approval_id in self.approval_cache:
                    del self.approval_cache[approval_id]
                logging.info(f"Upload completed for {file_name}.")

        except Exception as e:
            logging.error(f"Error processing upload : {e}")
            raise (e)

    async def check_and_finalize_upload(self, upload_id: str, total_chunks: int, minio_path: str):
        """
        Checks if all chunks are uploaded and finalizes the multipart upload.
        """
        try:
            uploaded_chunks = await self.minio.get_uploaded_parts(minio_path, upload_id)
            
            if len(uploaded_chunks) == total_chunks:
                logging.debug("Finalizing upload in MinIO...")
                
                # Finalize the upload
                response = await self.minio.complete_multipart_upload(minio_path, upload_id)
                
                logging.info("Upload finalized.")
                return response

        except Exception as e:
            logging.error(f"Error finalizing upload: {e}")

        return False


    async def update_postgres(self, minio_path: str, user_id: str, file_name: str, file_size: int, etag: str, mime_type: str):

        try:

            logging.debug("Inserting file metadata into PostgreSQL...")

            await self.db.insert_file_metadata(
                user_id=user_id,
                file_name=file_name,
                file_path=minio_path,
                file_size=file_size,
                file_hash=etag,
                mime_type=mime_type,
            )

            logging.info("Successfully updated PostgreSQL.")

        except Exception as e:
            logging.error(f"Failed to update PostgreSQL: {e}")
            raise (e)