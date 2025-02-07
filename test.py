import asyncio
import os
import logging
import hashlib
import aiofiles

from app.services.upload_file.upload_request_receiver import UploadRequestReceiver
from dotenv import load_dotenv

load_dotenv()

# Load Configurations from .env
MINIO_CONFIG = {
    "endpoint": os.getenv("MINIO_ENDPOINT"),
    "access_key": os.getenv("MINIO_ACCESS_KEY"),
    "secret_key": os.getenv("MINIO_SECRET_KEY"),
}

DB_URL = os.getenv("DATABASE_URL")
REDIS_URL = os.getenv("REDIS_URL")

# Initialize Upload Request Receiver
upload_receiver = UploadRequestReceiver(MINIO_CONFIG, DB_URL, REDIS_URL)

# File to be uploaded
TEST_FILE_PATH = "ragas_paper.pdf"  # Replace with actual file
USER_ID = "user-56789"
RELATIVE_PATH = "uploads"  # Folder where file is stored inside MinIO
CHUNK_SIZE = 5 * 1024 * 1024  # 5MB per chunk


async def compute_file_hash(file_path):
    """Computes SHA256 hash of the file for integrity verification."""
    sha256 = hashlib.sha256()
    async with aiofiles.open(file_path, "rb") as f:
        while chunk := await f.read(CHUNK_SIZE):
            sha256.update(chunk)
    return sha256.hexdigest()


async def simulate_upload():
    """
    Simulates a frontend uploading a file in chunks and sending it to MinIO via our backend services.
    """
    # Step 1: Read File and Compute Metadata
    file_size = os.path.getsize(TEST_FILE_PATH)
    file_name = os.path.basename(TEST_FILE_PATH)
    total_chunks = (file_size // CHUNK_SIZE) + (1 if file_size % CHUNK_SIZE != 0 else 0)

    # Step 2: Send Metadata for Validation
    metadata_payload = {
        "user_id": USER_ID,
        "file_name": file_name,
        "relative_path": RELATIVE_PATH,
        "file_size": file_size,
        "mime_type": "application/pdf",
        "total_chunks": total_chunks,
    }

    validation_response = await upload_receiver.receive_upload_request(metadata_payload)
    print(validation_response)
    
    if not validation_response["success"]:
        logging.error("File validation failed.")
        return
    
    upload_id = validation_response["upload_id"]
    uploadapproval_id = validation_response["uploadapproval_id"]

    print(f"✅ File approved for upload. Upload ID: {upload_id}, Approval ID: {uploadapproval_id}")

    # Step 3: Upload File in Chunks
    async with aiofiles.open(TEST_FILE_PATH, "rb") as f:
        for chunk_number in range(1, total_chunks + 1):
            chunk_data = await f.read(CHUNK_SIZE)

            chunk_payload = {
                "user_id": USER_ID,
                "file_name": file_name,
                "relative_path": RELATIVE_PATH,
                "file_size": file_size,
                "mime_type": "text/plain",
                "upload_id": upload_id,
                "uploadapproval_id": uploadapproval_id,
                "total_chunks": total_chunks,
                "chunk_number": chunk_number,
                "file_data": chunk_data,
            }

            upload_response = await upload_receiver.receive_upload_request(chunk_payload, chunk_data)
            print(upload_response)

            if not upload_response["success"]:
                logging.error(f"Chunk {chunk_number} failed to upload.")
                return

            print(f"✅ Uploaded chunk {chunk_number}/{total_chunks}")

    print(f"✅ File {file_name} uploaded successfully in {total_chunks} chunks.")

    # Step 4: Check Metadata in PostgreSQL (Optional Verification)
    stored_metadata = await upload_receiver.validator.db.get_file_metadata(USER_ID, file_name)
    if stored_metadata:
        print(f"✅ File metadata stored in PostgreSQL: {stored_metadata.file_path}")
    else:
        print("⚠ Metadata not found in PostgreSQL.")

# Run the test
asyncio.run(simulate_upload())