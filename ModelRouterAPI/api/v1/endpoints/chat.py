import asyncio
import json
import time
import uuid
import tiktoken
from typing import List, Dict, Any, Optional, Union

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from db.session import get_db
from core.security import get_api_key
from db.models import ApiKey, Usage
from schemas.chat import ChatCompletionRequest, ChatCompletionResponse, ChatCompletionChoice, ChatMessage
from schemas.chat import ChatCompletionChunkResponse, ChatCompletionChunkChoice, ChatCompletionChunkDelta, UsageInfo

# Import our model handlers and model selector
from model_handler import ModelRouter
from model_type import ModelType
from core.model_selector import ModelSelector
from config import settings

router = APIRouter()

# Helper function to estimate tokens (very rough estimation)
def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Count the number of tokens in a text string."""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except:
        # Fallback: rough approximation if tiktoken doesn't support the model
        return len(text) // 4  # Very rough approximation

async def generate_chat_response(model_router: ModelRouter, messages: List[ChatMessage]):
    # Convert messages to a prompt format understood by the model
    if model_router.provider.value == "openai":
        # OpenAI already understands the chat format
        prompt = messages
    else:
        # For other providers, convert to a simple text format
        prompt = ""
        for msg in messages:
            role_prefix = {
                "system": "System: ",
                "user": "User: ",
                "assistant": "Assistant: ",
                "function": "Function: ",
            }.get(msg.role, f"{msg.role.capitalize()}: ")
            prompt += f"{role_prefix}{msg.content}\n"
        
        # Add assistant prefix for the response
        prompt += "Assistant: "
    
    # Generate text with the model
    response = await model_router.generate_text(prompt)
    return response

@router.post("/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(
    request: ChatCompletionRequest,
    background_tasks: BackgroundTasks,
    api_key: ApiKey = Depends(get_api_key),
    db: Session = Depends(get_db),
):
    """
    Creates a model response for the given chat conversation.
    Uses intelligent routing based on model name patterns.
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    try:
        # Calculate prompt tokens
        estimated_token_count = sum(count_tokens(msg.content, request.model) for msg in request.messages)
        
        # Initialize the model router using the intelligent routing factory
        model_router = ModelRouter.initialize_from_model_name(
            model_name=request.model,
            model_type=ModelType.TEXT_GENERATION,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            stream=False,
        )
        
        # Generate the response
        response_text = await generate_chat_response(model_router, request.messages)
        
        # Calculate completion tokens
        completion_tokens = count_tokens(response_text, request.model)
        total_tokens = estimated_token_count + completion_tokens
        
        # Log usage
        completion_time = time.time() - start_time
        background_tasks.add_task(
            log_usage,
            db=db,
            api_key_id=getattr(api_key, "id", None),
            request_id=request_id,
            endpoint="chat.completions",
            model=request.model,
            provider=model_router.provider.value,
            prompt_tokens=estimated_token_count,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            processing_time=completion_time,
            request_data=request.json()
        )
        
        # Create response
        chat_response = ChatCompletionResponse(
            id=f"chatcmpl-{request_id}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=response_text),
                    finish_reason="stop"
                )
            ],
            usage=UsageInfo(
                prompt_tokens=estimated_token_count,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens
            )
        )
        
        return chat_response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/completions/stream")
async def create_chat_completion_stream(
    request: ChatCompletionRequest,
    background_tasks: BackgroundTasks,
    api_key: ApiKey = Depends(get_api_key),
    db: Session = Depends(get_db),
):
    """
    Creates a streaming chat completion for the given conversation.
    """
    if not request.stream:
        # If stream is not requested, redirect to non-streaming endpoint
        return await create_chat_completion(request, background_tasks, api_key, db)
    
    async def generate_stream():
        start_time = time.time()
        request_id = str(uuid.uuid4())
        created = int(time.time())
        
        try:
            # Calculate prompt tokens
            estimated_token_count = sum(count_tokens(msg.content, request.model) for msg in request.messages)
            
            # Initialize model using the intelligent routing factory
            model_router = ModelRouter.initialize_from_model_name(
                model_name=request.model,
                model_type=ModelType.TEXT_GENERATION,
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
                stream=True,
            )
            
            # Send initial chunk with role
            initial_chunk = ChatCompletionChunkResponse(
                id=f"chatcmpl-{request_id}",
                created=created,
                model=request.model,
                choices=[
                    ChatCompletionChunkChoice(
                        index=0,
                        delta=ChatCompletionChunkDelta(role="assistant"),
                        finish_reason=None
                    )
                ]
            )
            yield f"data: {json.dumps(initial_chunk.dict())}\n\n"
            
            # Generate streaming response
            accumulated_text = ""
            
            # Convert messages to prompt format
            if model_router.provider.value == "openai":
                prompt = request.messages
            else:
                # For other providers, convert to text format
                prompt = ""
                for msg in request.messages:
                    role_prefix = {
                        "system": "System: ",
                        "user": "User: ",
                        "assistant": "Assistant: ",
                        "function": "Function: "
                    }.get(msg.role, f"{msg.role.capitalize()}: ")
                    prompt += f"{role_prefix}{msg.content}\n"
                prompt += "Assistant: "
            
            # Generate streaming response
            async for text_chunk in await model_router.generate_text(prompt, stream=True):
                accumulated_text += text_chunk
                
                chunk = ChatCompletionChunkResponse(
                    id=f"chatcmpl-{request_id}",
                    created=created,
                    model=request.model,
                    choices=[
                        ChatCompletionChunkChoice(
                            index=0,
                            delta=ChatCompletionChunkDelta(content=text_chunk),
                            finish_reason=None
                        )
                    ]
                )
                yield f"data: {json.dumps(chunk.dict())}\n\n"
                await asyncio.sleep(0.01)  # Small delay to avoid overwhelming the client
            
            # Send final chunk with finish_reason
            final_chunk = ChatCompletionChunkResponse(
                id=f"chatcmpl-{request_id}",
                created=created,
                model=request.model,
                choices=[
                    ChatCompletionChunkChoice(
                        index=0,
                        delta=ChatCompletionChunkDelta(),
                        finish_reason="stop"
                    )
                ]
            )
            yield f"data: {json.dumps(final_chunk.dict())}\n\n"
            yield "data: [DONE]\n\n"
            
            # Calculate completion tokens
            completion_tokens = count_tokens(accumulated_text, request.model)
            total_tokens = estimated_token_count + completion_tokens
            
            # Log usage
            completion_time = time.time() - start_time
            background_tasks.add_task(
                log_usage,
                db=db,
                api_key_id=getattr(api_key, "id", None),
                request_id=request_id,
                endpoint="chat.completions.stream",
                model=request.model,
                provider=model_router.provider.value,
                prompt_tokens=estimated_token_count,
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