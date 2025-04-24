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
        
        response_text = ""
        # Check if we need structured output (JSON schema)
        if request.response_format and request.response_format.type == "json_schema":
            # Get the schema from the response_format
            schema = {}
            if request.response_format.json_schema:
                # Handle both direct schema and nested schema formats
                if isinstance(request.response_format.json_schema, dict) and "schema" in request.response_format.json_schema:
                    # OpenAI format with nested schema
                    schema = request.response_format.json_schema["schema"]
                else:
                    # Direct schema format
                    schema = request.response_format.json_schema
                
            # Create a prompt for structured output generation
            prompt_text = "\n".join([f"{msg.role}: {msg.content}" for msg in request.messages])
            
            # Generate structured output
            structured_output = await model_router.generate_structured_output(
                prompt=prompt_text,
                schema=schema,
                max_tokens=max_tokens
            )
            
            # Convert the structured output to JSON string
            response_text = json.dumps(structured_output)
        
        # Handle regular JSON object format
        elif request.response_format and request.response_format.type == "json_object":
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
            
            # Generate response
            response_text = await model_router.generate_text(
                prompt=request.messages,
                max_tokens=max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=request.stop
            )
        else:
            # Standard text generation
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
                    message=ChatMessage(
                        role="assistant", 
                        content=response_text,
                        refusal=None,
                        annotations=[]
                    ),
                    finish_reason="stop",
                    logprobs=None
                )
            ],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                prompt_tokens_details={
                    "cached_tokens": 0,
                    "audio_tokens": 0
                },
                completion_tokens_details={
                    "reasoning_tokens": 0,
                    "audio_tokens": 0,
                    "accepted_prediction_tokens": 0,
                    "rejected_prediction_tokens": 0
                }
            ),
            system_fingerprint=system_fingerprint,
            service_tier="default",
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
    """Generate streaming chat completions in OpenAI-compatible format"""
    start_time = time.time()
    request_id = str(uuid.uuid4())
    created_time = int(time.time())
    system_fingerprint = f"fp_{uuid.uuid4().hex[:10]}"
    
    # Calculate tokens for input
    prompt_tokens = sum(count_tokens(msg.content or "", request.model) for msg in request.messages)
    accumulated_text = ""
    
    try:
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
        
        # Send initial chunk with role AND empty content to match OpenAI format
        initial_chunk = ChatCompletionChunkResponse(
            id=f"chatcmpl-{request_id}",
            created=created_time,
            model=request.model,
            system_fingerprint=system_fingerprint,
            service_tier="default",  # Match OpenAI's format
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=ChatCompletionChunkDelta(
                        role="assistant",
                        content="",
                        refusal=None
                    ),
                    finish_reason=None,
                    logprobs=None  # Include logprobs field set to null to match OpenAI
                )
            ],
            object="chat.completion.chunk"
        )
        yield f"data: {initial_chunk.model_dump_json(exclude_none=False)}\n\n"
        
        # Handle different response formats
        if request.response_format and request.response_format.type == "json_schema":
            # Get the schema from the response_format
            schema = {}
            if request.response_format.json_schema:
                # Handle both direct schema and nested schema formats
                if isinstance(request.response_format.json_schema, dict) and "schema" in request.response_format.json_schema:
                    # OpenAI format with nested schema
                    schema = request.response_format.json_schema["schema"]
                else:
                    # Direct schema format
                    schema = request.response_format.json_schema
                
            # Create a prompt for structured output generation
            prompt_text = "\n".join([f"{msg.role}: {msg.content}" for msg in request.messages])
            
            # For structured JSON schema output, we need to generate the full response first
            # as we can't easily stream partial JSON while ensuring it's valid
            try:
                structured_output = await model_router.generate_structured_output(
                    prompt=prompt_text,
                    schema=schema,
                    max_tokens=max_tokens
                )
                
                # Convert the structured output to JSON string
                response_text = json.dumps(structured_output)
                accumulated_text = response_text
                
                # Stream the JSON as a sequence of characters for compatibility
                # Stream in small chunks to simulate streaming
                chunk_size = 10  # Adjust as needed
                for i in range(0, len(response_text), chunk_size):
                    text_chunk = response_text[i:i+chunk_size]
                    chunk = ChatCompletionChunkResponse(
                        id=f"chatcmpl-{request_id}",
                        created=created_time,
                        model=request.model,
                        system_fingerprint=system_fingerprint,
                        service_tier="default",
                        choices=[
                            ChatCompletionChunkChoice(
                                index=0,
                                delta=ChatCompletionChunkDelta(content=text_chunk, refusal=None),
                                finish_reason=None,
                                logprobs=None
                            )
                        ],
                        object="chat.completion.chunk"
                    )
                    yield f"data: {chunk.model_dump_json(exclude_none=False)}\n\n"
                    await asyncio.sleep(0.01)
            except Exception as e:
                error_chunk = {
                    "error": {
                        "message": f"Error generating structured output: {str(e)}",
                        "type": "server_error"
                    }
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"
        
        # Handle regular JSON object format
        elif request.response_format and request.response_format.type == "json_object":
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
                        service_tier="default",
                        choices=[
                            ChatCompletionChunkChoice(
                                index=0,
                                delta=ChatCompletionChunkDelta(content=text_chunk, refusal=None),
                                finish_reason=None,
                                logprobs=None
                            )
                        ],
                        object="chat.completion.chunk"
                    )
                    yield f"data: {chunk.model_dump_json(exclude_none=False)}\n\n"
                    await asyncio.sleep(0.01)
        
        # Standard text generation
        else:
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
                        service_tier="default",
                        choices=[
                            ChatCompletionChunkChoice(
                                index=0,
                                delta=ChatCompletionChunkDelta(content=text_chunk, refusal=None),
                                finish_reason=None,
                                logprobs=None
                            )
                        ],
                        object="chat.completion.chunk"
                    )
                    yield f"data: {chunk.model_dump_json(exclude_none=False)}\n\n"
                    await asyncio.sleep(0.01)
        
        # Send final chunk with finish_reason
        final_chunk = ChatCompletionChunkResponse(
            id=f"chatcmpl-{request_id}",
            created=created_time,
            model=request.model,
            system_fingerprint=system_fingerprint,
            service_tier="default",
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=ChatCompletionChunkDelta(refusal=None),  # Empty delta with refusal=null
                    finish_reason="stop",
                    logprobs=None
                )
            ],
            object="chat.completion.chunk"
        )
        yield f"data: {final_chunk.model_dump_json(exclude_none=False)}\n\n"
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