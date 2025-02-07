import asyncio
import logging
from io import BytesIO
from minio.error import S3Error
from app.config import settings
from app.core.storage_bin.minio.session_minio import minio_session

class MinIOHandler:
    """
    Handles file storage operations with MinIO, including file upload, download, and multipart upload.
    Updated to accept a pre-configured Minio client and bucket name.
    """
    def __init__(self, client, bucket_name):
        self.client = minio_session.client
        if self.client is None:
            logging.error("MinIO client is not initialized. Check session_minio connection in main.py.")
        self.bucket_name = settings.MINIO_BUCKET_NAME
        asyncio.create_task(self.ensure_bucket_exists())

    async def ensure_bucket_exists(self):
        """
        Ensures that the MinIO bucket exists. If not, creates the bucket.
        Wrapped with asyncio.to_thread if the underlying method is blocking.
        """
        try:
            exists = await asyncio.to_thread(self.client.bucket_exists, self.bucket_name)
            if not exists:
                await asyncio.to_thread(self.client.make_bucket, self.bucket_name)
                logging.info(f"Bucket '{self.bucket_name}' created successfully.")
        except S3Error as e:
            logging.error(f"Error ensuring MinIO bucket exists: {e}")
            raise Exception("Error creating MinIO bucket")
        
    def abort_incomplete_upload(self, object_name: str):
        """
        For testing purposes: aborts any incomplete multipart uploads.
        """
        incomplete_uploads = self.client._list_multipart_uploads(bucket_name=self.bucket_name, prefix=object_name)
        count = 0
        for upload in incomplete_uploads.uploads:
            object_name = upload.object_name
            upload_id = upload.upload_id
            count += 1
            print(count, ": upload object_name=", upload.object_name, "\nupload id =", upload.upload_id)
            try:
                self.client._abort_multipart_upload(
                    self.bucket_name,
                    object_name,
                    upload_id
                )
            except S3Error as e:
                logging.error(f"Aborted incomplete upload ID={upload.upload_id}")
                print("error", e)
                raise e
        print("abort completed")
        incomplete = self.client._list_multipart_uploads(bucket_name=self.bucket_name, prefix=object_name)
        count = 0
        for upload in incomplete.uploads:
            count += 1
        print("Count of incomplete uploads after aborting: ", count)

    async def start_multipart_upload(self, object_path: str) -> str:
        """
        Initializes a multipart upload and returns the upload ID.
        """
        try:
            self.abort_incomplete_upload(object_name=object_path)
            upload_id = await asyncio.to_thread(
                self.client._create_multipart_upload,
                bucket_name=self.bucket_name,
                object_name=object_path,
                headers={}
            )
            logging.info(f"Multipart upload started for {object_path} with ID: {upload_id}")
            return upload_id
        except S3Error as e:
            logging.error(f"Initiating the Multipart upload failed: {e}")
            raise Exception("Initiating the Multipart upload failed")
    
    async def upload_part(self, minio_path, upload_id, chunk_number, file_data: BytesIO):
        """
        Uploads a specific part of a file in a multipart upload session.
        """
        try:
            response = await asyncio.to_thread(
                self.client._upload_part,
                bucket_name=self.bucket_name,
                object_name=minio_path,
                data=file_data,
                part_number=chunk_number,
                upload_id=upload_id,
                headers={}
            )
            logging.info(f"Uploaded part {chunk_number} for {minio_path}")
            return {"part_number": chunk_number, "etag": response}
        except S3Error as e:
            logging.error(f"Uploading the part failed: {e}")
            return False
        
    async def complete_multipart_upload(self, object_name: str, upload_id: str):
        """
        Completes a multipart upload by combining all uploaded parts.
        """
        try:
            parts_response = await asyncio.to_thread(
                self.client.list_parts, self.bucket_name, object_name, upload_id
            )
            parts = parts_response.uploads if hasattr(parts_response, "uploads") else []
            sorted_parts = sorted(parts, key=lambda part: part.part_number)
            await asyncio.to_thread(
                self.client._complete_multipart_upload,
                bucket_name=self.bucket_name,
                object_name=object_name,
                upload_id=upload_id,
                parts=sorted_parts
            )
            logging.info(f"Multipart upload completed for '{object_name}', Upload ID: {upload_id}")
        except S3Error as e:
            logging.error(f"Completing the multipart upload failed: {e}")
            raise Exception("Completing the multipart upload failed")
    
    async def abort_multipart_upload(self, object_name: str, upload_id: str):
        """
        Aborts an incomplete multipart upload session.
        """
        try:
            await asyncio.to_thread(
                self.client._abort_multipart_upload,
                self.bucket_name,
                object_name,
                upload_id
            )
            logging.info(f"Multipart upload aborted for '{object_name}', Upload ID: {upload_id}")
        except S3Error as e:
            logging.error(f"Aborting the multipart upload failed: {e}")
            raise Exception("Aborting the multipart upload failed")
        
    async def get_uploaded_parts(self, object_name: str, upload_id: str):
        """
        Fetches the list of uploaded parts for a multipart upload session.
        """
        try:
            parts_response = await asyncio.to_thread(
                self.client.list_parts, self.bucket_name, object_name, upload_id
            )
            parts = parts_response.uploads if hasattr(parts_response, "uploads") else []
            sorted_parts = sorted(parts, key=lambda part: part.part_number)
            logging.info(f"Retrieved {len(sorted_parts)} parts for multipart upload session.")
            return sorted_parts
        except S3Error as e:
            logging.error(f"Fetching uploaded parts failed: {e}")
            raise Exception("Fetching uploaded parts failed")