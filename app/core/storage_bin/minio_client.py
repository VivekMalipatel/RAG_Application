from minio import Minio
import os
from dotenv import load_dotenv

load_dotenv()

class MinIOClient:
    def __init__(self):
        self.client = Minio(
            endpoint=os.getenv("MINIO_ENDPOINT", "localhost:9000"),
            access_key=os.getenv("MINIO_ROOT_USER"),
            secret_key=os.getenv("MINIO_ROOT_PASSWORD"),
            secure=False
        )
        self.bucket_prefix = os.getenv("MINIO_BUCKET_NAME", "omnirag-storage")

    async def create_user_bucket(self, user_id: str):
        """Asynchronously creates a bucket for the user if it doesn't exist."""
        bucket_name = f"{self.bucket_prefix}-{user_id}".lower()
        
        if not self.client.bucket_exists(bucket_name):
            self.client.make_bucket(bucket_name)
            print(f"Created bucket: {bucket_name}")
        return bucket_name

    async def upload_file(self, user_id: str, file_data: bytes, file_name: str):
        """Uploads a file asynchronously to MinIO."""
        bucket_name = await self.create_user_bucket(user_id)
        object_name = f"{user_id}/{file_name}"
        
        with open(file_name, "wb") as f:
            f.write(file_data)  # Temporarily store for MinIO upload
        
        self.client.fput_object(bucket_name, object_name, file_name)
        os.remove(file_name)  # Clean up temp file
        
        return f"{bucket_name}/{object_name}"

    async def get_file(self, user_id: str, file_name: str):
        """Retrieves a file asynchronously from MinIO."""
        bucket_name = f"{self.bucket_prefix}-{user_id}".lower()
        local_path = f"downloads/{file_name}"

        self.client.fget_object(bucket_name, f"{user_id}/{file_name}", local_path)
        return local_path