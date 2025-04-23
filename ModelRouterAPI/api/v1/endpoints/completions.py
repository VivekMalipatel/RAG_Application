import asyncio
import json
import time
import uuid
from typing import List, Dict, Any, Optional, Union

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
import tiktoken

from db.session import get_db
from core.security import get_api_key
from db.models import ApiKey, Usage
from schemas.completions import CompletionRequest, CompletionResponse, CompletionChoice
from schemas.completions import CompletionChunkResponse, CompletionChunkChoice
from schemas.chat import UsageInfo

# Import our model handlers
from model_handler import ModelRouter
from model_provider import Provider
from model_type import ModelType

router = APIRouter()

# Helper function to count tokens - reused from chat.py
def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Count the number of tokens in a text string."""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except:
        # Fallback: rough approximation if tiktoken doesn't support the model
        return len(text) // 4  # Very rough approximation

@router.post("", response_model=CompletionResponse)
async def create_completion(
    request: CompletionRequest,
    background_tasks: BackgroundTasks,
    api_key: ApiKey = Depends(get_api_key),
    db: Session = Depends(get_db),
):
    """
    Creates a completion for the provided prompt and parameters.
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    try:
        # Handle streaming requests
        if request.stream:
            return await create_completion_stream(request, background_tasks, api_key, db)
        
        # Convert prompt to string if it's an array
        prompt = request.prompt[0] if isinstance(request.prompt, list) else request.prompt
        
        # Parse model info to determine the provider
        model_parts = request.model.split("/")
        if len(model_parts) > 1 and model_parts[0] in ["mistralai", "meta-llama", "google"]:
            provider = Provider.HUGGINGFACE
        elif request.model.startswith(("gpt-", "text-")):
            provider = Provider.OPENAI
        else:
            provider = Provider.OLLAMA
        
        # Initialize the appropriate model
        model_router = ModelRouter(
            provider=provider,
            model_name=request.model,
            model_type=ModelType.TEXT_GENERATION,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            stream=False,
        )
        
        # Calculate prompt tokens
        prompt_tokens = count_tokens(prompt, request.model)
        
        # Generate the response
        response_text = await model_router.generate_text(prompt)
        
        # If echo is true, prepend the prompt to the response
        if request.echo:
            response_text = prompt + response_text
        
        # Calculate completion tokens
        completion_tokens = count_tokens(response_text, request.model)
        total_tokens = prompt_tokens + completion_tokens
        
        # Log usage
        completion_time = time.time() - start_time
        background_tasks.add_task(
            log_usage,
            db=db,
            api_key_id=getattr(api_key, "id", None),
            request_id=request_id,
            endpoint="completions",
            model=request.model,
            provider=provider.value,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            processing_time=completion_time,
            request_data=request.json()
        )
        
        # Create response
        completion_response = CompletionResponse(
            id=f"cmpl-{request_id}",
            created=int(time.time()),
            model=request.model,
            choices=[
                CompletionChoice(
                    text=response_text,
                    index=0,
                    logprobs=None,
                    finish_reason="stop"
                )
            ],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens
            )
        )
        
        return completion_response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/stream")
async def create_completion_stream(
    request: CompletionRequest,
    background_tasks: BackgroundTasks,
    api_key: ApiKey = Depends(get_api_key),
    db: Session = Depends(get_db),
):
    """
    Creates a streaming completion for the provided prompt and parameters.
    """
    if not request.stream:
        # If stream is not requested, redirect to non-streaming endpoint
        return await create_completion(request, background_tasks, api_key, db)
    
    async def generate_stream():
        start_time = time.time()
        request_id = str(uuid.uuid4())
        created = int(time.time())
        
        try:
            # Convert prompt to string if it's an array
            prompt = request.prompt[0] if isinstance(request.prompt, list) else request.prompt
            
            # Similar setup as non-streaming endpoint
            model_parts = request.model.split("/")
            if len(model_parts) > 1 and model_parts[0] in ["mistralai", "meta-llama", "google"]:
                provider = Provider.HUGGINGFACE
            elif request.model.startswith(("gpt-", "text-")):
                provider = Provider.OPENAI
            else:
                provider = Provider.OLLAMA
            
            model_router = ModelRouter(
                provider=provider,
                model_name=request.model,
                model_type=ModelType.TEXT_GENERATION,
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
                stream=True,
            )
            
            # Calculate prompt tokens
            prompt_tokens = count_tokens(prompt, request.model)
            
            # If echo is true, we need to send the prompt first
            if request.echo:
                initial_chunk = CompletionChunkResponse(
                    id=f"cmpl-{request_id}",
                    created=created,
                    model=request.model,
                    choices=[
                        CompletionChunkChoice(
                            text=prompt,
                            index=0,
                            logprobs=None,
                            finish_reason=None
                        )
                    ]
                )
                yield f"data: {json.dumps(initial_chunk.dict())}\n\n"
            
            # Generate streaming response
            accumulated_text = ""
            
            # Generate streaming response
            async for text_chunk in await model_router.generate_text(prompt, stream=True):
                accumulated_text += text_chunk
                
                chunk = CompletionChunkResponse(
                    id=f"cmpl-{request_id}",
                    created=created,
                    model=request.model,
                    choices=[
                        CompletionChunkChoice(
                            text=text_chunk,
                            index=0,
                            logprobs=None,
                            finish_reason=None
                        )
                    ]
                )
                yield f"data: {json.dumps(chunk.dict())}\n\n"
                await asyncio.sleep(0.01)  # Small delay to avoid overwhelming the client
            
            # Send final chunk with finish_reason
            final_chunk = CompletionChunkResponse(
                id=f"cmpl-{request_id}",
                created=created,
                model=request.model,
                choices=[
                    CompletionChunkChoice(
                        text="",
                        index=0,
                        logprobs=None,
                        finish_reason="stop"
                    )
                ]
            )
            yield f"data: {json.dumps(final_chunk.dict())}\n\n"
            yield "data: [DONE]\n\n"
            
            # Calculate completion tokens
            completion_tokens = count_tokens(accumulated_text, request.model)
            total_tokens = prompt_tokens + completion_tokens
            
            # Log usage
            completion_time = time.time() - start_time
            background_tasks.add_task(
                log_usage,
                db=db,
                api_key_id=getattr(api_key, "id", None),
                request_id=request_id,
                endpoint="completions.stream",
                model=request.model,
                provider=provider.value,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                processing_time=completion_time,
                request_data=request.json()
            )
            
        except Exception as e:
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

# Helper function to log usage - same as in other endpoint files
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