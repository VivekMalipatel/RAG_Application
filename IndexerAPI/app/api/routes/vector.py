from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File, Form
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, HttpUrl
import json
import aiohttp
import hashlib

from app.services.vector_store import VectorStore
from app.core.model.model_handler import ModelHandler
from app.processors.file_processor import FileProcessor
import logging
import base64
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.database import get_db

logger = logging.getLogger(__name__)

router = APIRouter()
file_processor = FileProcessor()

class SearchTextRequest(BaseModel):
    text: str
    top_k: int = 10

class SearchImageRequest(BaseModel):
    image_base64: str
    text: Optional[str] = None
    top_k: int = 10

class SearchBatchTextRequest(BaseModel):
    texts: List[str]
    top_k: int = 10

class ProcessPdfUrlRequest(BaseModel):
    url: HttpUrl
    source: str
    metadata: Optional[Dict[str, Any]] = None

@router.post("/search/text", response_model=List[List[Dict[str, Any]]])
async def search_by_text(request: SearchTextRequest, req: Request):
    try:
        model_handler = req.app.state.model_handler
        vector_store = req.app.state.vector_store
        query_embeddings = await model_handler.embed_text([request.text])
        results = vector_store.search(
            query_vectors=query_embeddings,
            k=request.top_k
        )
        
        return results
    except Exception as e:
        logger.error(f"Error during text search: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

@router.post("/search/image", response_model=List[List[Dict[str, Any]]])
async def search_by_image(request: SearchImageRequest, req: Request):
    try:
        model_handler = req.app.state.model_handler
        vector_store = req.app.state.vector_store
        image_text = request.text or "Image query"
        query_data = [{"image": request.image_base64, "text": image_text}]
        
        query_embeddings = await model_handler.embed_image(query_data)
        
        results = vector_store.search(
            query_vectors=query_embeddings,
            k=request.top_k
        )
        
        return results
    except Exception as e:
        logger.error(f"Error during image search: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")
    
@router.post("/search/text/batch", response_model=List[List[Dict[str, Any]]])
async def search_text_batch(request: SearchBatchTextRequest, req: Request):
    try:
        model_handler = req.app.state.model_handler
        vector_store = req.app.state.vector_store
        query_embeddings = await model_handler.embed_text(request.texts)
        results = vector_store.search_batch(
            queries=query_embeddings,
            k=request.top_k
        )
        return results
    except Exception as e:
        logger.error(f"Error during batch text search: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch search error: {str(e)}")

@router.get("/stats")
async def get_vector_stats(req: Request):
    vector_store = req.app.state.vector_store
    return vector_store.get_stats()

@router.delete("/documents/{doc_id}", response_model=Dict[str, Any])
async def delete_document_from_store(doc_id: str, req: Request):
    logger.info(f"Attempting to remove document: {doc_id} from vector store.")
    try:
        vector_store = req.app.state.vector_store
        removed = vector_store.remove_document(doc_id)
        if removed:
            saved = vector_store.save()
            if saved:
                logger.info(f"Successfully removed document {doc_id} and saved the index.")
                return {"message": f"Document {doc_id} and its vectors removed successfully.", "doc_id": doc_id, "status": "removed"}
            else:
                logger.error(f"Removed document {doc_id}, but failed to save the index.")
                raise HTTPException(status_code=500, detail=f"Document {doc_id} removed from memory, but failed to save index.")
        else:
            logger.warning(f"Document {doc_id} not found in vector store for removal.")
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found.")
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Error during document removal for doc_id {doc_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error removing document: {str(e)}")

@router.post("/process/pdf/url", response_model=Dict[str, Any])
async def process_pdf_from_url(
    request: ProcessPdfUrlRequest,
    db: AsyncSession = Depends(get_db)
):
    logger.info(f"Received URL for PDF processing from source: {request.source}")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(str(request.url)) as response:
                if response.status != 200:
                    raise HTTPException(status_code=response.status, 
                                       detail=f"Failed to fetch file: {response.reason}")
                file_content = await response.read()
        
        result = await file_processor.process(file_content, request.metadata)
        
        return result
    except aiohttp.ClientError as e:
        logger.error(f"Error downloading file: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error downloading file: {str(e)}")
    except Exception as e:
        logger.error(f"Error during file processing: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"File processing error: {str(e)}")

@router.post("/process/pdf", response_model=Dict[str, Any])
async def process_pdf(
    file: UploadFile = File(...),
    source: str = Form(...),
    metadata: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db)
):
    logger.info(f"Received file for processing: {file.filename} from source: {source}")
    
    meta_dict = None
    if metadata:
        try:
            meta_dict = json.loads(metadata)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid metadata format")
    
    #TODO: Validate file type and size
    file_content = await file.read()
    
    try:
        result = await file_processor.process(file_content, meta_dict)
        
        return result
    except Exception as e:
        logger.error(f"Error during file processing: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"File processing error: {str(e)}")