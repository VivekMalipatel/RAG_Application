import asyncio
import json
import time
import uuid
import tiktoken
from typing import List, Dict, Any, Optional, Union, AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from db.session import get_db
from core.security import get_api_key
from db.models import ApiKey, Usage
from schemas.chat import (
    ChatCompletionRequest, ChatCompletionResponse, ChatCompletionChoice, ChatMessage,
    ChatCompletionChunkResponse, ChatCompletionChunkChoice, ChatCompletionChunkDelta, UsageInfo
)

# Import model handlers
from model_handler import ModelRouter
from model_type import ModelType

router = APIRouter()

def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Count the number of tokens in a text string"""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception:
        # Fallback approximation - ensure we return an integer
        return int(len(text.split()) * 1.3)

@router.post("/chat/completions", response_model=None)
async def create_chat_completion(
    request: ChatCompletionRequest,
    background_tasks: BackgroundTasks,
    api_key: ApiKey = Depends(get_api_key),
    db: Session = Depends(get_db)
):
    """Create a model response for chat conversations"""
    # Handle streaming request
    if request.stream:
        return StreamingResponse(
            generate_chat_stream(request, background_tasks, api_key, db),
            media_type="text/event-stream"
        )
    
    # Handle non-streaming request
    start_time = time.time()
    request_id = str(uuid.uuid4())
    created_time = int(time.time())
    system_fingerprint = f"fp_{uuid.uuid4().hex[:10]}"
    
    try:
        # Calculate tokens for input
        prompt_tokens = sum(count_tokens(msg.content or "", request.model) for msg in request.messages)
        
        # Handle JSON response format if requested
        if request.response_format and request.response_format.type == "json_object":
            # Ensure there's a system message with JSON instructions
            system_msg = next((msg for msg in request.messages if msg.role == "system"), None)
            
            if system_msg:
                system_msg.content = f"{system_msg.content or ''}\nRespond with JSON format only."
            else:
                # Add system message for JSON format
                request.messages.insert(0, ChatMessage(
                    role="system", 
                    content="Respond with JSON format only."
                ))
        
        # Initialize model router with request parameters
        max_tokens = request.max_tokens
        model_router = await ModelRouter.initialize_from_model_name(
            model_name=request.model,
            model_type=ModelType.TEXT_GENERATION,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=max_tokens,
            frequency_penalty=request.frequency_penalty,
            presence_penalty=request.presence_penalty,
            stop=request.stop
        )
        
        # Generate response - no need to format messages as our clients now handle all formats
        response_text = await model_router.generate_text(
            prompt=request.messages,
            max_tokens=max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=request.stop
        )
        
        # Calculate tokens for output
        completion_tokens = count_tokens(response_text, request.model)
        total_tokens = prompt_tokens + completion_tokens
        
        # Log usage to database
        completion_time = time.time() - start_time
        background_tasks.add_task(
            log_usage,
            db=db,
            api_key_id=getattr(api_key, "id", None),
            request_id=request_id,
            endpoint="chat/completions",
            model=request.model,
            provider=model_router.provider.value,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            processing_time=completion_time,
            request_data=request.model_dump_json()
        )
        
        # Create response in OpenAI format
        return ChatCompletionResponse(
            id=f"chatcmpl-{request_id}",
            created=created_time,
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=response_text),
                    finish_reason="stop"
                )
            ],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens
            ),
            system_fingerprint=system_fingerprint,
            object="chat.completion"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def generate_chat_stream(
    request: ChatCompletionRequest,
    background_tasks: BackgroundTasks,
    api_key: ApiKey,
    db: Session
) -> AsyncGenerator[str, None]:
    """Generate streaming chat completions"""
    start_time = time.time()
    request_id = str(uuid.uuid4())
    created_time = int(time.time())
    system_fingerprint = f"fp_{uuid.uuid4().hex[:10]}"
    
    # Calculate tokens for input
    prompt_tokens = sum(count_tokens(msg.content or "", request.model) for msg in request.messages)
    accumulated_text = ""
    
    try:
        # Handle JSON response format
        if request.response_format and request.response_format.type == "json_object":
            # Ensure there's a system message with JSON instructions
            system_msg = next((msg for msg in request.messages if msg.role == "system"), None)
            
            if system_msg:
                system_msg.content = f"{system_msg.content or ''}\nRespond with JSON format only."
            else:
                # Add system message for JSON format
                request.messages.insert(0, ChatMessage(
                    role="system", 
                    content="Respond with JSON format only."
                ))
        
        # Initialize model router
        max_tokens = request.max_tokens
        model_router = await ModelRouter.initialize_from_model_name(
            model_name=request.model,
            model_type=ModelType.TEXT_GENERATION,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=max_tokens,
            frequency_penalty=request.frequency_penalty,
            presence_penalty=request.presence_penalty,
            stop=request.stop,
            stream=True
        )
        
        # Send initial chunk with role
        initial_chunk = ChatCompletionChunkResponse(
            id=f"chatcmpl-{request_id}",
            created=created_time,
            model=request.model,
            system_fingerprint=system_fingerprint,
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=ChatCompletionChunkDelta(role="assistant"),
                    finish_reason=None
                )
            ],
            object="chat.completion.chunk"
        )
        yield f"data: {initial_chunk.model_dump_json(exclude_unset=True)}\n\n"
        
        try:
            # Generate streaming response
            stream_generator = await model_router.generate_text(
                prompt=request.messages,
                max_tokens=max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=request.stop,
                stream=True
            )
            
            async for text_chunk in stream_generator:
                # Skip empty chunks
                if text_chunk:
                    accumulated_text += text_chunk
                    
                    # Stream chunk in OpenAI format
                    chunk = ChatCompletionChunkResponse(
                        id=f"chatcmpl-{request_id}",
                        created=created_time,
                        model=request.model,
                        system_fingerprint=system_fingerprint,
                        choices=[
                            ChatCompletionChunkChoice(
                                index=0,
                                delta=ChatCompletionChunkDelta(content=text_chunk),
                                finish_reason=None
                            )
                        ],
                        object="chat.completion.chunk"
                    )
                    yield f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n"
                    # Small delay to control stream flow
                    await asyncio.sleep(0.01)
            
        except Exception as e:
            # Handle streaming errors
            error_chunk = {
                "error": {
                    "message": f"Error during streaming: {str(e)}",
                    "type": "server_error"
                }
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
        
        # Send final chunk with finish_reason
        final_chunk = ChatCompletionChunkResponse(
            id=f"chatcmpl-{request_id}",
            created=created_time,
            model=request.model,
            system_fingerprint=system_fingerprint,
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=ChatCompletionChunkDelta(),  # Empty delta
                    finish_reason="stop"
                )
            ],
            object="chat.completion.chunk"
        )
        yield f"data: {final_chunk.model_dump_json(exclude_unset=True)}\n\n"
        yield "data: [DONE]\n\n"
        
        # Calculate completion tokens and log usage
        completion_tokens = count_tokens(accumulated_text, request.model)
        total_tokens = prompt_tokens + completion_tokens
        
        # Log usage to database
        completion_time = time.time() - start_time
        background_tasks.add_task(
            log_usage,
            db=db,
            api_key_id=getattr(api_key, "id", None),
            request_id=request_id,
            endpoint="chat/completions",
            model=request.model,
            provider=model_router.provider.value,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            processing_time=completion_time,
            request_data=request.model_dump_json()
        )
        
    except Exception as e:
        # Handle setup errors
        error_chunk = {
            "error": {
                "message": str(e),
                "type": "server_error"
            }
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
        yield "data: [DONE]\n\n"

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
    """Log API usage to database"""
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
        db.rollback()