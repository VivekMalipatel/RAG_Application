from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File, Form
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, HttpUrl
import json
import aiohttp

from app.services.vector_store import VectorStore
from app.core.model.model_handler import ModelHandler
from app.processors.file_processor import FileProcessor
import logging
import base64
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.database import get_db

logger = logging.getLogger(__name__)

router = APIRouter()
vector_store = VectorStore()
vector_store.load()
model_handler = ModelHandler()
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

@router.post("/search/text", response_model=List[Dict[str, Any]])
async def search_by_text(request: SearchTextRequest):
    try:
        query_embeddings = await model_handler.embed_text([request.text])
        
        results = vector_store.search(
            query_vectors=query_embeddings,
            k=request.top_k
        )
        
        return results
    except Exception as e:
        logger.error(f"Error during text search: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

@router.post("/search/image", response_model=List[Dict[str, Any]])
async def search_by_image(request: SearchImageRequest):
    try:
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
async def search_text_batch(request: SearchBatchTextRequest):
    try:
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
async def get_vector_stats():
    return vector_store.get_stats()

@router.post("/process/pdf/url", response_model=Dict[str, Any])
async def process_pdf_from_url(
    request: ProcessPdfUrlRequest,
    db: AsyncSession = Depends(get_db)
):
    logger.info(f"Received URL for PDF processing from source: {request.source}")
    
    try:
        # Download file from the signed URL
        async with aiohttp.ClientSession() as session:
            async with session.get(str(request.url)) as response:
                if response.status != 200:
                    raise HTTPException(status_code=response.status, 
                                       detail=f"Failed to fetch file: {response.reason}")
                file_content = await response.read()
        
        # Process the file using FileProcessor
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
    # The File Type and Size will be depending on what files and size the Markdown class can handle.
    
    # Read file content
    file_content = await file.read()
    
    try:
        # Process the file using FileProcessor
        # Fix: Pass parameters as expected by the process method
        result = await file_processor.process(file_content, meta_dict)
        
        return result
    except Exception as e:
        logger.error(f"Error during file processing: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"File processing error: {str(e)}")