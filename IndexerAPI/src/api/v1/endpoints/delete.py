import logging
from fastapi import APIRouter, HTTPException
from schemas.schemas import DeleteFileRequest, DeleteResponse
from core.storage.neo4j_handler import get_neo4j_handler
from core.storage.s3_handler import build_document_s3_base_path, get_global_s3_handler

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/file", response_model=DeleteResponse)
async def delete_file(request: DeleteFileRequest):
    logger.info(f"Received file delete request for: {request.filename} from user: {request.user_id}")
    try:
        neo4j_handler = get_neo4j_handler()
        s3_handler = await get_global_s3_handler()
        s3_base_path = build_document_s3_base_path(request.org_id, request.user_id, request.source, request.filename)
        s3_prefixes = [s3_base_path, f"metadata/{s3_base_path}"]
        s3_deleted = True
        for prefix in s3_prefixes:
            result = await s3_handler.delete_prefix(prefix)
            if not result:
                s3_deleted = False
        success = await neo4j_handler.delete_document(user_id=request.user_id, org_id=request.org_id, source=request.source, filename=request.filename)
        if success and s3_deleted:
            return DeleteResponse(success=True, message=f"File '{request.filename}' deleted successfully")
        return DeleteResponse(success=False, message=f"Failed to delete file '{request.filename}'")
    except Exception as exc:
        logger.error(f"Error deleting file: {exc}")
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {exc}")
