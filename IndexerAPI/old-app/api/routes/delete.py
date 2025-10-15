import logging
from fastapi import APIRouter, HTTPException
from schemas.schemas import DeleteFileRequest, DeleteResponse
from core.storage.neo4j_handler import get_neo4j_handler

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/file", response_model=DeleteResponse)
async def delete_file(
    request: DeleteFileRequest,
):
    logger.info(f"Received file delete request for: {request.filename} from user: {request.user_id}")
    
    try:
        neo4j_handler = get_neo4j_handler()
        
        success = await neo4j_handler.delete_document(
            user_id=request.user_id,
            org_id=request.org_id,
            source=request.source,
            filename=request.filename
        )
        
        if success:
            return DeleteResponse(
                success=True,
                message=f"File '{request.filename}' deleted successfully"
            )
        else:
            return DeleteResponse(
                success=False,
                message=f"Failed to delete file '{request.filename}'"
            )
        
    except Exception as e:
        logger.error(f"Error deleting file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {str(e)}")