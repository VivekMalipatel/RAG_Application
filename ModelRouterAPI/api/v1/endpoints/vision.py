import asyncio
import base64
import json
import time
import uuid
import os
import tempfile
import httpx
from typing import List, Dict, Any, Optional, Union, Tuple
from urllib.parse import urlparse
from io import BytesIO
from PIL import Image

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, File, UploadFile, Form, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from db.session import get_db
from core.security import get_api_key
from db.models import ApiKey, Usage
from schemas.vision import VisionRequest, VisionResponse, VisionChoice, VisionMessage
from schemas.vision import VisionChunkResponse, VisionChunkChoice, VisionChunkDelta
from schemas.chat import UsageInfo, ChatMessage

from model_handler import ModelRouter
from model_provider import Provider
from model_type import ModelType
from core.model_selector import ModelSelector, TaskType
from config import settings

router = APIRouter()
model_selector = ModelSelector()

async def download_image(image_url: str) -> BytesIO:
    """Download an image from a URL and return it as BytesIO object."""
    async with httpx.AsyncClient() as client:
        response = await client.get(image_url)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail=f"Failed to download image: HTTP {response.status_code}")
        return BytesIO(response.content)

async def save_image_to_temp_file(image_data: BytesIO) -> str:
    """Save an image to a temporary file and return the file path."""
    # Create a temporary file with a .jpg extension
    fd, path = tempfile.mkstemp(suffix='.jpg')
    os.close(fd)
    
    # Save the image
    with Image.open(image_data) as img:
        img.save(path)
    
    return path

async def process_content_for_provider(
    content: Union[str, List[Dict[str, Any]]], 
    provider: Provider
) -> Tuple[str, List[str]]:
    """
    Process multimodal content for different providers.
    Returns the formatted prompt and a list of temporary file paths to clean up later.
    """
    temp_files = []
    
    # Handle string content (text only)
    if isinstance(content, str):
        return content, temp_files
    
    # Handle complex content (text + images)
    formatted_prompt = ""
    
    for item in content:
        if item["type"] == "text":
            formatted_prompt += f"{item.get('text', '')}\n"
        
        elif item["type"] == "image_url":
            image_url_data = item.get("image_url", {})
            if isinstance(image_url_data, dict):
                image_url = image_url_data.get("url", "")
            else:
                # Handle case where image_url is not a dict
                continue
                
            # Handle base64 encoded images
            if image_url.startswith("data:image"):
                # Extract the base64 part
                base64_data = image_url.split(",", 1)[1]
                image_data = BytesIO(base64.b64decode(base64_data))
            else:
                # Download remote image
                image_data = await download_image(image_url)
            
            # Different handling based on provider
            if provider == Provider.OPENAI:
                # OpenAI can handle the URL or base64 directly in API call
                pass  # No additional processing needed
            
            elif provider == Provider.OLLAMA:
                # Ollama needs a temporary file path
                tmp_path = await save_image_to_temp_file(image_data)
                temp_files.append(tmp_path)
                formatted_prompt += f"<image>{tmp_path}</image>\n"
                
            elif provider == Provider.HUGGINGFACE:
                # For HuggingFace, we can either encode the image description or use a placeholder
                formatted_prompt += "[IMAGE CONTENT]\n"
    
    return formatted_prompt, temp_files

@router.post("", response_model=VisionResponse)
async def create_vision_completion(
    request: VisionRequest,
    background_tasks: BackgroundTasks,
    auto_select: bool = Query(None),  # Override config setting
    api_key: ApiKey = Depends(get_api_key),
    db: Session = Depends(get_db),
):
    """
    Creates a response for the given multimodal prompt with text and images.
    """
    # Check if vision is enabled in settings
    if not settings.ENABLE_VISION:
        raise HTTPException(status_code=400, detail="Vision API is not enabled on this server")
        
    start_time = time.time()
    request_id = str(uuid.uuid4())
    temp_files = []
    
    try:
        # Determine if we should auto-select model based on content
        use_auto_select = auto_select if auto_select is not None else settings.AUTO_SELECT_MODEL
        
        # Extract text content for task type detection
        user_content = ""
        for msg in request.messages:
            if msg.role == "user":
                if isinstance(msg.content, str):
                    user_content += msg.content + " "
                elif isinstance(msg.content, list):
                    for item in msg.content:
                        if item.get("type") == "text":
                            user_content += item.get("text", "") + " "
        
        system_prompt = next((msg.content if isinstance(msg.content, str) else "" 
                             for msg in request.messages if msg.role == "system"), None)
        
        # Get token estimate (rough approximation as images aren't counted)
        estimated_token_count = len(user_content.split()) * 1.5  # Rough approximation
        
        if use_auto_select:
            # Always use image understanding task type for vision requests
            task_type = TaskType.IMAGE_UNDERSTANDING
            
            # Select the best model
            provider_name, model_name = model_selector.select_best_model(
                task_type=task_type,
                model_type=ModelType.MULTIMODAL,
                token_count=int(estimated_token_count),
                preferred_model=request.model if request.model != "auto" else None
            )
            
            # Convert provider name to enum
            if provider_name == "openai":
                provider = Provider.OPENAI
            elif provider_name == "ollama": 
                provider = Provider.OLLAMA
            else:
                provider = Provider.HUGGINGFACE
        else:
            # Use model specified in the request
            model_name = request.model
            
            # Determine provider based on model name
            if model_name.startswith(("gpt-4-vision", "gpt-4v", "gpt-4o", "claude-3")):
                provider = Provider.OPENAI
            elif model_name in ["llava", "bakllava"]:
                provider = Provider.OLLAMA
            else:
                provider = Provider.HUGGINGFACE
        
        # Initialize the appropriate model
        model_router = ModelRouter(
            provider=provider,
            model_name=model_name,
            model_type=ModelType.MULTIMODAL,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            stream=False,
        )
        
        # Process messages differently based on provider capabilities
        chat_messages = []
        
        for msg in request.messages:
            # Process complex content
            processed_content, new_temp_files = await process_content_for_provider(
                msg.content, provider
            )
            temp_files.extend(new_temp_files)
            
            # For OpenAI, we can pass the original message structure
            if provider == Provider.OPENAI:
                chat_messages.append(msg)
            else:
                # For other providers, use the processed content
                chat_messages.append(ChatMessage(
                    role=msg.role,
                    content=processed_content,
                    name=msg.name
                ))
        
        # Generate response using the appropriate provider
        if provider == Provider.OPENAI:
            response_text = await model_router.generate_text(chat_messages)
        else:
            # For other providers, convert to a simple text format
            prompt = ""
            for msg in chat_messages:
                role_prefix = {
                    "system": "System: ",
                    "user": "User: ",
                    "assistant": "Assistant: ",
                }.get(msg.role, f"{msg.role.capitalize()}: ")
                prompt += f"{role_prefix}{msg.content}\n"
            
            # Add assistant prefix for the response
            prompt += "Assistant: "
            response_text = await model_router.generate_text(prompt)
        
        # Estimate completion tokens
        completion_tokens = len(response_text.split()) * 1.5  # Rough approximation
        total_tokens = estimated_token_count + completion_tokens
        
        # Log usage
        completion_time = time.time() - start_time
        background_tasks.add_task(
            log_usage,
            db=db,
            api_key_id=getattr(api_key, "id", None),
            request_id=request_id,
            endpoint="vision",
            model=model_name,
            provider=provider.value,
            prompt_tokens=int(estimated_token_count),
            completion_tokens=int(completion_tokens),
            total_tokens=int(total_tokens),
            processing_time=completion_time,
            request_data=request.json()
        )
        
        # Clean up temporary files
        background_tasks.add_task(cleanup_temp_files, temp_files)
        
        # Create response
        vision_response = VisionResponse(
            id=f"chatcmpl-{request_id}",
            created=int(time.time()),
            model=model_name,
            choices=[
                VisionChoice(
                    index=0,
                    message=VisionMessage(role="assistant", content=response_text),
                    finish_reason="stop"
                )
            ],
            usage=UsageInfo(
                prompt_tokens=int(estimated_token_count),
                completion_tokens=int(completion_tokens),
                total_tokens=int(total_tokens)
            )
        )
        
        return vision_response
    
    except Exception as e:
        # Clean up temporary files in case of error
        cleanup_temp_files(temp_files)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/stream")
async def create_vision_completion_stream(
    request: VisionRequest,
    background_tasks: BackgroundTasks,
    auto_select: bool = Query(None),  # Override config setting
    api_key: ApiKey = Depends(get_api_key),
    db: Session = Depends(get_db),
):
    """
    Creates a streaming vision completion for multimodal prompts.
    """
    # Check if vision is enabled in settings
    if not settings.ENABLE_VISION:
        raise HTTPException(status_code=400, detail="Vision API is not enabled on this server")
        
    if not request.stream:
        # If stream is not requested, redirect to non-streaming endpoint
        return await create_vision_completion(request, background_tasks, auto_select, api_key, db)
    
    async def generate_stream():
        start_time = time.time()
        request_id = str(uuid.uuid4())
        created = int(time.time())
        temp_files = []
        
        try:
            # Determine if we should auto-select model based on content
            use_auto_select = auto_select if auto_select is not None else settings.AUTO_SELECT_MODEL
            
            # Extract text content for task type detection
            user_content = ""
            for msg in request.messages:
                if msg.role == "user":
                    if isinstance(msg.content, str):
                        user_content += msg.content + " "
                    elif isinstance(msg.content, list):
                        for item in msg.content:
                            if item.get("type") == "text":
                                user_content += item.get("text", "") + " "
            
            system_prompt = next((msg.content if isinstance(msg.content, str) else "" 
                                for msg in request.messages if msg.role == "system"), None)
            
            # Get token estimate (rough approximation as images aren't counted)
            estimated_token_count = len(user_content.split()) * 1.5  # Rough approximation
            
            if use_auto_select:
                # Always use image understanding task type for vision requests
                task_type = TaskType.IMAGE_UNDERSTANDING
                
                # Select the best model
                provider_name, model_name = model_selector.select_best_model(
                    task_type=task_type,
                    model_type=ModelType.MULTIMODAL,
                    token_count=int(estimated_token_count),
                    preferred_model=request.model if request.model != "auto" else None
                )
                
                # Convert provider name to enum
                if provider_name == "openai":
                    provider = Provider.OPENAI
                elif provider_name == "ollama": 
                    provider = Provider.OLLAMA
                else:
                    provider = Provider.HUGGINGFACE
            else:
                # Use model specified in the request
                model_name = request.model
                
                # Determine provider based on model name
                if model_name.startswith(("gpt-4-vision", "gpt-4v", "gpt-4o", "claude-3")):
                    provider = Provider.OPENAI
                elif model_name in ["llava", "bakllava"]:
                    provider = Provider.OLLAMA
                else:
                    provider = Provider.HUGGINGFACE
            
            # Initialize the appropriate model
            model_router = ModelRouter(
                provider=provider,
                model_name=model_name,
                model_type=ModelType.MULTIMODAL,
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
                stream=True,
            )
            
            # Process messages differently based on provider capabilities
            chat_messages = []
            
            for msg in request.messages:
                # Process complex content
                processed_content, new_temp_files = await process_content_for_provider(
                    msg.content, provider
                )
                temp_files.extend(new_temp_files)
                
                # For OpenAI, we can pass the original message structure
                if provider == Provider.OPENAI:
                    chat_messages.append(msg)
                else:
                    # For other providers, use the processed content
                    chat_messages.append(ChatMessage(
                        role=msg.role,
                        content=processed_content,
                        name=msg.name
                    ))
            
            # Send initial chunk with role
            initial_chunk = VisionChunkResponse(
                id=f"chatcmpl-{request_id}",
                created=created,
                model=model_name,
                choices=[
                    VisionChunkChoice(
                        index=0,
                        delta=VisionChunkDelta(role="assistant"),
                        finish_reason=None
                    )
                ]
            )
            yield f"data: {json.dumps(initial_chunk.dict())}\n\n"
            
            # Generate streaming response
            accumulated_text = ""
            
            if provider == Provider.OPENAI:
                prompt = chat_messages
            else:
                # For other providers, convert to a simple text format
                prompt = ""
                for msg in chat_messages:
                    role_prefix = {
                        "system": "System: ",
                        "user": "User: ",
                        "assistant": "Assistant: ",
                    }.get(msg.role, f"{msg.role.capitalize()}: ")
                    prompt += f"{role_prefix}{msg.content}\n"
                
                # Add assistant prefix for the response
                prompt += "Assistant: "
            
            # Generate streaming response
            async for text_chunk in await model_router.generate_text(prompt, stream=True):
                accumulated_text += text_chunk
                
                chunk = VisionChunkResponse(
                    id=f"chatcmpl-{request_id}",
                    created=created,
                    model=model_name,
                    choices=[
                        VisionChunkChoice(
                            index=0,
                            delta=VisionChunkDelta(content=text_chunk),
                            finish_reason=None
                        )
                    ]
                )
                yield f"data: {json.dumps(chunk.dict())}\n\n"
                await asyncio.sleep(0.01)  # Small delay to avoid overwhelming the client
            
            # Send final chunk with finish_reason
            final_chunk = VisionChunkResponse(
                id=f"chatcmpl-{request_id}",
                created=created,
                model=model_name,
                choices=[
                    VisionChunkChoice(
                        index=0,
                        delta=VisionChunkDelta(),
                        finish_reason="stop"
                    )
                ]
            )
            yield f"data: {json.dumps(final_chunk.dict())}\n\n"
            yield "data: [DONE]\n\n"
            
            # Calculate completion tokens
            completion_tokens = len(accumulated_text.split()) * 1.5  # Rough approximation
            total_tokens = estimated_token_count + completion_tokens
            
            # Log usage
            completion_time = time.time() - start_time
            background_tasks.add_task(
                log_usage,
                db=db,
                api_key_id=getattr(api_key, "id", None),
                request_id=request_id,
                endpoint="vision.stream",
                model=model_name,
                provider=provider.value,
                prompt_tokens=int(estimated_token_count),
                completion_tokens=int(completion_tokens),
                total_tokens=int(total_tokens),
                processing_time=completion_time,
                request_data=request.json()
            )
            
            # Clean up temporary files
            background_tasks.add_task(cleanup_temp_files, temp_files)
            
        except Exception as e:
            # Clean up temporary files in case of error
            cleanup_temp_files(temp_files)
            error_chunk = {
                "error": {
                    "message": str(e),
                    "type": "server_error"
                }
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream"
    )

def cleanup_temp_files(file_paths: List[str]):
    """Clean up temporary files."""
    for path in file_paths:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception as e:
            print(f"Error removing temporary file {path}: {str(e)}")

# Helper function to log usage
def log_usage(
    db: Session, 
    api_key_id: Optional[int],
    request_id: str,
    endpoint: str,
    model: str,
    provider: str,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    processing_time: float,
    request_data: str
):
    """Log API usage to database for tracking and billing."""
    try:
        usage_record = Usage(
            api_key_id=api_key_id,
            timestamp=time.time(),
            endpoint=endpoint,
            model=model,
            provider=provider,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            processing_time=processing_time,
            request_id=request_id,
            request_data=request_data
        )
        db.add(usage_record)
        db.commit()
    except Exception as e:
        # Log error but don't fail the request
        print(f"Error logging usage: {str(e)}")
        db.rollback()