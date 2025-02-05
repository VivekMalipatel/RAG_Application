import os
import asyncio
import logging
import aiofiles
from io import BytesIO
from minio import Minio
from minio.error import S3Error
from dotenv import load_dotenv

load_dotenv()

class MinIOHandler:
    """handles file storage operations with MinIO, 
    including file upload, download, and multipart upload."""
    def __init__(self, endpoint: str, access_key: str, secret_key: str, bucket_name: str, secure: bool = False):
        """
        Initializes the MinIO client with configurable parameters.

        Args:
            endpoint (str): MinIO server URL (e.g., 'localhost:9000').
            access_key (str): MinIO username or access key.
            secret_key (str): MinIO password or secret key.
            bucket_name (str): The bucket name to be used.
            secure (bool, optional): Whether to use HTTPS. Defaults to True.
        """
        self.client = Minio(
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure
        )
        self.bucket_name = bucket_name
        self.ensure_bucket_exists()
    
    def ensure_bucket_exists(self):
        """
        Ensures that the MinIO bucket exists. If not, it creates the bucket.
        """
        try:
            if not self.client.bucket_exists(self.bucket_name):
                self.client.make_bucket(self.bucket_name)
                print(f"Bucket '{self.bucket_name}' created successfully.")
        except S3Error as e:
            print(f"Error ensuring MinIO bucket exists: {e}")
            raise


    async def start_multipart_upload(self, object_path: str) -> str:
        """
        Initializes a multipart upload and returns the upload ID.

        Args:
            object_path (str): The destination object path in MinIO.

        Returns:
            str: The upload ID if successful, else None.
        """
        try:
            upload_id = self.client.create_multipart_upload(self.bucket_name, object_path)
            print(f"Multipart upload started for {object_path} with ID: {upload_id}")
            return upload_id
        except S3Error as e:
            print(f"Initiating the Multipart upload have been failed\n: {e}")
            return None
            
    
    async def upload_part(self, minio_path, upload_id, chunk_number, file_data):
        """
        # Upload a specific part of a file in a multipart upload session.

        Args:
            minio_path (str): The destination object path in MinIO.
            upload_id (str): The multipart upload session ID.
            chunk_number (int): The part number.
            file_data (BytesIO): The file data to be uploaded.
        """
        try:
            file_stream = BytesIO(file_data)
            response = await asyncio.to_thread(
                self.client.put_object,
                self.bucket_name,
                minio_path,
                file_data,
                length=len(file_data),
                part_number=chunk_number,
                upload_id=upload_id
            )
            logging.info("Uploaded part {chunk_number} for {minio_path}")
            return {"part_number": chunk_number, "etag": response.etag}
        except S3Error as e:
            logging.error("Uploading the part have been failed\n: {e}")
            return False
        
    async def complete_multipart_upload(self, object_name: str, upload_id: str, parts: list):
        """
        Completes a multipart upload by combining all uploaded parts.

        Args:
            object_name (str): The destination object path in MinIO.
            upload_id (str): The multipart upload session ID.
            parts (list): List of uploaded parts.

        Returns:
            str: The final object path in MinIO if successful, else None.
        """
        try:
            await asyncio.to_thread(
                self.client.complete_multipart_upload,
                self.bucket_name,
                object_name,
                upload_id,
                parts
            )
            logging.info(f"Multipart upload completed for '{object_name}', Upload ID: {upload_id}")
            return f"{self.bucket_name}/{object_name}"
        except S3Error as e:
            logging.error(f"Error completing multipart upload for '{object_name}': {e}")
            return None
    
    async def abort_multipart_upload(self, object_name: str, upload_id: str):
        """
        Aborts an incomplete multipart upload session.

        Args:
            object_name (str): The destination object path in MinIO.
            upload_id (str): The multipart upload session ID.
        """
        try:
            # List all active multipart uploads for the bucket and object
            uploads = self.client.list_multipart_uploads(self.bucket_name, prefix=object_name)
            for upload in uploads:
                if upload.key == object_name and upload.upload_id == upload_id:
                    logging.info(f"Found active multipart upload for {object_name} with ID: {upload_id}")
                    break
        except S3Error as e:
            logging.error(f"upload ID havent been found while aborting the multipart upload\n : {e}")
            return False
        
        # Abort the multipart upload
        try:
            await asyncio.to_thread(
                self.client.abort_multipart_upload,
                self.bucket_name,
                object_name,
                upload_id
            )
            logging.info(f"Multipart upload aborted successfully for {object_name}")
            return True
        except S3Error as e:
            logging.error(f"Error aborting multipart upload for {object_name}: {e}")
            return False