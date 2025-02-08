import os
import requests
import json
import time
import base64

# ✅ API Endpoints
UPLOAD_APPROVAL_URL = "http://localhost:8000/files/upload/"
USER_ID = 1234324

# ✅ File Details
FILE_PATH = "Temp_Files/docs/ragas_paper.pdf"
FILE_NAME = "ragas_paper.pdf"
CHUNK_SIZE = 5 * 1024 * 1024  # 5MB per chunk

def get_upload_approval(file_name, total_chunks, file_size, mime_type):
    """
    Requests an upload approval ID from the API.
    """
    payload = {
        "user_id": USER_ID,
        "file_name": file_name,
        "local_file_path": FILE_PATH,
        "relative_path": "uploads",
        "mime_type": mime_type,
        "total_chunks": total_chunks,
        "file_size": file_size,
        "upload_id": None,
        "approval_id": None,
        "chunk_number": None
    }
    response = requests.post(UPLOAD_APPROVAL_URL, data={"request_data": json.dumps(payload)})
    response = json.loads(response.json()[0])

    return response.get("approval_id"), response.get("upload_id")


def upload_chunk(upload_id, approval_id, chunk_number, total_chunks, file_data, file_size):
    """
    Uploads a file chunk to the server.
    """
    encoded_chunk = base64.b64encode(file_data).decode('utf-8')
    payload = {
        "user_id": USER_ID,
        "approval_id": approval_id,
        "upload_id": upload_id,
        "chunk_number": chunk_number,
        "total_chunks": total_chunks,
        "relative_path": "uploads",
        "file_size": file_size,
        "mime_type": "application/pdf",
        "file_name": FILE_NAME,
        "local_file_path": FILE_PATH
    }
    files = {"file_data":encoded_chunk}
    response = requests.post(UPLOAD_APPROVAL_URL, data={"request_data": json.dumps(payload)}, files=files)

    if response.ok:
        print(f"✅ Chunk {chunk_number}/{total_chunks} uploaded successfully.")
    else:
        print(f"❌ Failed to upload chunk {chunk_number}: {response.json()}")


def main():
    """
    Simulates the complete file upload in chunks.
    """
    # ✅ Step 1: Read File & Calculate Chunks
    file_size = os.path.getsize(FILE_PATH)
    total_chunks = (file_size // CHUNK_SIZE) + (1 if file_size % CHUNK_SIZE != 0 else 0)

    # ✅ Step 2: Request Upload Approval
    approval_id, upload_id = get_upload_approval(
        file_name=os.path.basename(FILE_PATH),
        total_chunks=total_chunks,
        file_size=file_size,
        mime_type="application/pdf"
    )
    
    if not approval_id or not upload_id:
        print("❌ Failed to get upload approval. Exiting.")
        return

    print(f"🆗 Upload approved. Upload ID: {upload_id}, Approval ID: {approval_id}")

    # ✅ Step 3: Split File and Upload Chunks
    with open(FILE_PATH, "rb") as file:
        chunk_number = 1
        while chunk := file.read(CHUNK_SIZE):
            upload_chunk(upload_id, approval_id, chunk_number, total_chunks, chunk, file_size)
            chunk_number += 1
            time.sleep(0.5)  # Simulating network delay

    print("🚀 File upload completed successfully!")


if __name__ == "__main__":
    main()