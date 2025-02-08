from app.config import settings
from dotenv import load_dotenv
import os

load_dotenv()
print("MinIO Access Key:", type(settings.MINIO_ACCESS_KEY))
print("MinIO Secret Key:", type(os.getenv("MINIO_ACCESS_KEY")))