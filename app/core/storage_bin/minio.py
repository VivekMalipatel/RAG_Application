from minio import Minio
from minio.error import S3Error
import logging
from dotenv import load_dotenv
import os

class MinIOClient:
    def __init__(self):
        load_dotenv()  # Load environment variables
        
        self.client = Minio(
            "localhost:9000",
            access_key=os.getenv('MINIO_ROOT_USER'),
            secret_key=os.getenv('MINIO_ROOT_PASSWORD'),
            secure=False
        )
        self.bucket_prefix = os.getenv('MINIO_BUCKET_NAME')

    def create_user_bucket(self, user_id: str):
        """Create per-user storage"""
        try:
            # Create bucket name with prefix to ensure valid S3 bucket name
            bucket_name = f"{self.bucket_prefix}-{user_id}".lower()
            
            if not self.client.bucket_exists(bucket_name):
                self.client.make_bucket(bucket_name)
                logging.info(f"Created bucket: {bucket_name}")
            else:
                logging.info(f"Bucket already exists: {bucket_name}")
            return True
        except S3Error as exc:
            logging.error(f"User bucket creation failed: {exc}")
            return False

    def list_buckets(self):
        """List all buckets - useful for debugging"""
        try:
            buckets = self.client.list_buckets()
            return [bucket.name for bucket in buckets]
        except S3Error as exc:
            logging.error(f"Failed to list buckets: {exc}")
            return []
