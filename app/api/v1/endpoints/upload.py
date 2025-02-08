from fastapi import APIRouter, HTTPException, status, UploadFile, File, Form
import base64
from app.services.upload_file.upload_request_receiver import UploadRequestReceiver
import json

router = APIRouter()

@router.post("/upload/")
async def upload_file(
    request_data: str = Form(...),
    file_data: UploadFile = File(None)
):
    # Validate payload has request_data
    try:
        request_data = json.loads(request_data)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Invalid request_data JSON format"
        )
    # Required keys for request_data
    required_keys = [
        "user_id",
        "file_name",
        "local_file_path",
        "relative_path",
        "mime_type",
        "file_size",
        "upload_id",
        "chunk_number",
        "approval_id",
        "total_chunks"
    ]
    for key in required_keys:
        if key not in request_data:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid Request")

    file_content = None
    if file_data:
        try:
            encoded_chunk = await file_data.read()
            file_content = base64.b64decode(encoded_chunk)
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, 
                detail="Invalid file_data encoding"
            )
    
    # Forward to UploadRequestReceiver
    receiver = UploadRequestReceiver()
    response = await receiver.receive_upload_request(request_data, file_content)
    if response.get("success"):
        return json.dumps(response), status.HTTP_200_OK
    else:
        return json.dumps(response), status.HTTP_500_INTERNAL_SERVER_ERROR