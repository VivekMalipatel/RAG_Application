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
    def __init__(self):
        self.client = minio_session.client
        if self.client is None:
            logging.error("MinIO client is not initialized. Check session_minio connection in main.py.")
        self.bucket_name = settings.MINIO_BUCKET_NAME
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(self.ensure_bucket_exists())
        else:
            loop.run_until_complete(self.ensure_bucket_exists())

    async def ensure_bucket_exists(self):
        """
        Ensures that the MinIO bucket exists. If not, creates the bucket.
        """
        try:
            # Check if bucket exists using list_buckets
            response = await self.client.list_buckets()
            bucket_names = [bucket["Name"] for bucket in response["Buckets"]]

            if self.bucket_name in bucket_names:
                logging.info(f"Bucket '{self.bucket_name}' already exists.")
                return

            # Create bucket if it does not exist
            await self.client.create_bucket(Bucket=self.bucket_name)
            logging.info(f"Bucket '{self.bucket_name}' created successfully.")

        except Exception as e:
            logging.error(f"Error ensuring bucket exists: {e}")
            raise Exception("Error creating MinIO bucket")

    async def start_multipart_upload(self, object_path: str) -> str:
        """
        Initializes a multipart upload and returns the upload ID.
        """
        try:
            await self.ensure_bucket_exists()
            response = await self.client.create_multipart_upload(
                            Bucket=self.bucket_name,
                            Key=object_path,
                        )

            upload_id = response.get("UploadId")
            logging.info(f"Multipart upload started for {object_path} with ID: {upload_id}")
            return upload_id
        except S3Error as e:
            logging.error(f"Initiating the multipart upload failed: {e}")
            raise Exception("Initiating the multipart upload failed")
    
    async def upload_part(self, minio_path, upload_id, chunk_number, file_data: BytesIO):
        """
        Uploads a specific part of a file in a multipart upload session.
        """
        try:
            # Await the async function directly
            response = await self.client.upload_part(
                Bucket=self.bucket_name,
                Key=minio_path,
                UploadId=upload_id,
                PartNumber=chunk_number,
                Body=file_data
            )
            logging.info(f"Uploaded part {chunk_number} for {minio_path}")
            return {"part_number": chunk_number, "etag": response["ETag"]}
        except S3Error as e:
            logging.error(f"Uploading the part failed: {e}")
            return False
        
    async def complete_multipart_upload(self, object_name: str, upload_id: str):
        """
        Completes a multipart upload by combining all uploaded parts.
        """
        try:
            # Fetch the list of uploaded parts
            parts_response = await self.client.list_parts(
                Bucket=self.bucket_name,
                Key=object_name,
                UploadId=upload_id
            )
            
            # Extract and filter parts to include only PartNumber and ETag
            parts = [
                {
                    'PartNumber': part['PartNumber'],
                    'ETag': part['ETag']
                }
                for part in parts_response.get("Parts", [])
            ]

            # Sort the parts by PartNumber
            sorted_parts = sorted(parts, key=lambda part: part['PartNumber'])
            
            # Complete the multipart upload
            response = await self.client.complete_multipart_upload(
                Bucket=self.bucket_name,
                Key=object_name,
                UploadId=upload_id,
                MultipartUpload={"Parts": sorted_parts}
            )
            
            logging.info(f"Multipart upload completed for '{object_name}', Upload ID: {upload_id}")
            return response
        except S3Error as e:
            logging.error(f"Completing the multipart upload failed: {e}")
            raise Exception("Completing the multipart upload failed")
    
    async def abort_multipart_upload(self, object_name: str, upload_id: str):
        """
        Aborts an incomplete multipart upload session.
        """
        try:
            await asyncio.to_thread(
                self.client.abort_multipart_upload,
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
            # Await the async call directly
            parts_response = await self.client.list_parts(
                Bucket=self.bucket_name,
                Key=object_name,
                UploadId=upload_id
            )
            # Extract and sort parts
            parts = parts_response.get("Parts", [])
            sorted_parts = sorted(parts, key=lambda part: part["PartNumber"])
            logging.info(f"Retrieved {len(sorted_parts)} parts for multipart upload session.")
            return sorted_parts
        except S3Error as e:
            logging.error(f"Fetching uploaded parts failed: {e}")
            raise Exception("Fetching uploaded parts failed")
    
    async def fetch_file_from_minio(self,file_path: str) -> BytesIO:
        """Fetches a file from MinIO using its path."""
        try:
            client = minio_session.client
            bucket_name = file_path.split('/')[0]  # Assuming bucket name is part of the path
            object_name = '/'.join(file_path.split('/')[1:])
            
            response = await client.get_object(Bucket=bucket_name, Key=object_name)
            streaming_body = response['Body']
        
            if streaming_body is None:
                raise ValueError("No data received from MinIO")
                
            # Read the entire content into memory
            data = await streaming_body.read()
            streaming_body.close()
            
            # Create a BytesIO object with the data
            file_data = BytesIO(data)
            return file_data
        except Exception as e:
            logging.error(f"Error fetching file from MinIO: {e}")
            raise
