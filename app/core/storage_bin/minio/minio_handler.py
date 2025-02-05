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
            
    async def upload_file_multipart(self, object_name: str, file_path: str, upload_id, part_size=5 * 1024 * 1024):
        
        """
        Checks whther the upload_id is valid or not, if not it will start a new multipart upload session."""

        try:
            # List all active multipart uploads for the bucket and object
            uploads = self.client.list_multipart_uploads(self.bucket_name, prefix=object_name)
            for upload in uploads:
                if upload.key == object_name and upload.upload_id == upload_id:
                    logging.info(f"Found active multipart upload for {object_name} with ID: {upload_id}")
                    break
        except S3Error as e:
            logging.error(f"upload ID havent been found while uploading the file \n : {e}")
            return None, None
        
        
        """
        Splits a file into multiple parts and uploads them asynchronously.
        Returns a list of uploaded parts with `part_number` and `etag`.
        """
        try:
            # Step 1: Start Multipart Upload
            logging.info(f"Started multipart upload for '{object_name}', Upload ID: {upload_id}")

            parts = []
            part_number = 1

            async with aiofiles.open(file_path, "rb") as f:
                while True:
                    file_data = await f.read(part_size)
                    if not file_data:
                        break

                    # Upload each part
                    part_info = await self.upload_part(object_name, upload_id, part_number, file_data)
                    if part_info:
                        parts.append(part_info)
                        part_number += 1

            logging.info(f"Uploaded {len(parts)} parts for '{object_name}'.")
            return upload_id, parts  # Returns `upload_id` and `parts`

        except S3Error as e:
            logging.error(f"Multipart upload failed for '{object_name}': {e}")
            return None, None

    async def upload_part(self, object_name: str, upload_id: str, part_number: int, file_data: bytes):
        """
        Uploads a single file part asynchronously.
        Returns a dictionary containing `part_number` and `etag`.
        """
        try:
            file_stream = BytesIO(file_data)
            response = await asyncio.to_thread(
                self.client.upload_part,
                self.bucket_name,
                object_name,
                upload_id,
                part_number,
                file_stream,
                len(file_data)
            )
            logging.info(f"Part {part_number} uploaded for '{object_name}', Upload ID: {upload_id}")
            return {"part_number": part_number, "etag": response.etag}

        except S3Error as e:
            logging.error(f"Error uploading part {part_number} for '{object_name}': {e}")
            return None

    def complete_file_multipart(self, object_name: str, upload_id: str, parts: list):
        """
        Completes the multipart upload using the uploaded parts.

        Args:
            object_name (str): The destination object path in MinIO.
            upload_id (str): The multipart upload session ID.
            parts (list): List of uploaded parts.
        """
        try:
            if not parts:
                raise ValueError("No parts available to complete upload.")

            self.client.complete_multipart_upload(self.bucket_name, object_name, upload_id, parts)
            logging.info(f"Completed multipart upload for '{object_name}', Upload ID: {upload_id}")

        except S3Error as e:
            logging.error(f"Error completing multipart upload for '{object_name}': {e}")
        
    
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