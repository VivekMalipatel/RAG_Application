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
    def __init__(self, endpoint: str, access_key: str, secret_key: str, secure: bool = False):
        """
        Initializes the MinIO client with configurable parameters.

        Args:
            endpoint (str): MinIO server URL (e.g., 'localhost:9000').
            access_key (str): MinIO username or access key.
            secret_key (str): MinIO password or secret key.
            secure (bool, optional): Whether to use HTTPS. Defaults to True.
        """
        try:
            self.client = Minio(
                endpoint=endpoint,
                access_key=access_key,
                secret_key=secret_key,
                secure=secure
            )
            self.bucket_name = "miniobucket"
            self.ensure_bucket_exists()
        except Exception as e:
            logging.error(f"Error initializing MinIO client: {e}")
            return None
    
    def ensure_bucket_exists(self):
        """
        Ensures that the MinIO bucket exists. If not, it creates the bucket.
        """
        try:
            if not self.client.bucket_exists(self.bucket_name):
                self.client.make_bucket(self.bucket_name)
                logging.info(f"Bucket '{self.bucket_name}' created successfully.")
                
        except S3Error as e:
            logging.error(f"Error ensuring MinIO bucket exists: {e}")
            raise Exception("Error creating MinIO bucket")
        
    #did for testing purposes
    def abort_incomplete_upload(self, object_name: str):
        incomplete_uploads = self.client._list_multipart_uploads(bucket_name=self. bucket_name, prefix=object_name)
        count = 0
        for upload in incomplete_uploads.uploads:
            object_name= upload.object_name
            upload_id= upload.upload_id
            count += 1
            print(count,": upload object_name=" ,upload.object_name ,"\nupload id =", upload.upload_id )
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
        incomplete = self.client._list_multipart_uploads(bucket_name=self. bucket_name, prefix=object_name)
        count = 0
        for upload in incomplete.uploads:
            count += 1
            #print("upload object_name=" ,upload.object_name ,"\nupload id =", upload.upload_id )
        print("Count of incomplete uploads after aborting: ", count)


    async def start_multipart_upload(self, object_path: str) -> str:
        """
        Initializes a multipart upload and returns the upload ID.

        Args:
            object_path (str): The destination object path in MinIO.

        Returns:
            str: The upload ID if successful, else None.
        """
        try:
            # Abort any incomplete multipart uploads for this object  #used for testing purposes
            print("Abort any incomplete multipart uploads for this object")
            self.abort_incomplete_upload(object_name= object_path)
            
            upload_id = self.client._create_multipart_upload(bucket_name = self.bucket_name, object_name =object_path, headers={})
            #logging.info(f"Multipart upload started for {object_path} with ID: {upload_id}")
            logging.info(f"Multipart upload started for {object_path} with ID")
            print(f"Multipart upload started for {object_path} with ID")
            return upload_id
        except S3Error as e:
            logging.error(f"Initiating the Multipart upload have been failed\n: {e}")
            raise Exception("Initiating the Multipart upload have been failed")
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

            response = await asyncio.to_thread(
                self.client._upload_part,
                bucket_name = self.bucket_name,
                object_name = minio_path,
                data = file_data,
                part_number = chunk_number,
                upload_id= upload_id,
                headers = {}
            )
            logging.info(f"Uploaded part {chunk_number} for {minio_path}")
            print("response=", response)
            return {"part_number": chunk_number, "etag": response}
        except S3Error as e:
            logging.error(f"Uploading the part have been failed\n: {e}")
            return False
        
    async def complete_multipart_upload(self, object_name: str, upload_id: str):
        """
        Completes a multipart upload by combining all uploaded parts.

        Args:
            object_name (str): The destination object path in MinIO.
            upload_id (str): The multipart upload session ID.
            parts (list): List of uploaded parts.

        Returns:
            str: The final object path in MinIO if successful, else None.
        """
        parts = self.client.list_parts(self.bucket_name, object_name, upload_id)
        try:
            await asyncio.to_thread(
                self.client._complete_multipart_upload,
                bucket_name= self.bucket_name,
                object_name= object_name,
                upload_id =upload_id,
                parts =parts
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
            Listuploads = self.client._list_multipart_uploads(bucket_name=self.bucket_name, prefix=object_name)
            for upload in Listuploads.uploads:
                if upload.object_name == object_name and upload.upload_id == upload_id:
                    logging.info(f"Found active multipart upload for {object_name} with ID: {upload_id}")
                    break
        except S3Error as e:
            logging.error(f"upload ID havent been found while aborting the multipart upload\n : {e}")
            return False
        
        # Abort the multipart upload
        try:
            await asyncio.to_thread(
                self.client._abort_multipart_upload,
                self.bucket_name,
                object_name,
                upload_id
            )
            logging.info(f"Multipart upload aborted successfully for {object_name}")
            return True
        except S3Error as e:
            logging.error(f"Error aborting multipart upload for {object_name}: {e}")
            return False
        
    async def get_uploaded_parts(self, object_name, upload_id):
        """
        Fetches the number of parts uploaded for a multipart upload session.

        Args:
            object_name (str): The destination object path in MinIO.
            upload_id (str): The multipart upload session ID.

        Returns:
            set: The number of parts uploaded if successful, else None.
        """
        try:           
            # List all parts uploaded for the object
            parts = self.client._list_parts(self.bucket_name, object_name, upload_id)
            return set(part.part_number for part in parts)
        except S3Error as e:
            logging.error(f"Error fetching uploaded parts for {object_name}: {e}")
            return None