import asyncio
import json
import time
import uuid
import tiktoken
import base64
from typing import List, Dict, Any, Optional, Union, AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from fastapi.responses import StreamingResponse, JSONResponse
from sqlalchemy.orm import Session

from db.session import get_db
from core.security import get_api_key
from db.models import ApiKey, Usage
from schemas.chat import (
    ChatCompletionRequest, ChatCompletionResponse, ChatCompletionChoice, ChatMessage,
    ChatCompletionChunkResponse, ChatCompletionChunkChoice, ChatCompletionChunkDelta, UsageInfo
)

from model_handler import ModelRouter
from model_type import ModelType

router = APIRouter()

def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception:
        if isinstance(text, str):
            return int(len(text.split()) * 1.3)
        else:
            return int(len(text) * 1.3)

@router.post("/chat/completions", response_model=None)
async def create_chat_completion(
    request: ChatCompletionRequest,
    background_tasks: BackgroundTasks,
    api_key: ApiKey = Depends(get_api_key),
    db: Session = Depends(get_db)
):
    if request.stream:
        return StreamingResponse(
            generate_chat_stream(request, background_tasks, api_key, db),
            media_type="text/event-stream"
        )
    
    start_time = time.time()
    request_id = str(uuid.uuid4())
    created_time = int(time.time())
    system_fingerprint = f"fp_{uuid.uuid4().hex[:10]}"
    
    try:
        prompt_tokens = sum(count_tokens(msg.content or "", request.model) for msg in request.messages)
        
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
        if request.response_format and request.response_format.type == "json_schema":
            schema = {}
            if request.response_format.json_schema:
                if isinstance(request.response_format.json_schema, dict) and "schema" in request.response_format.json_schema:
                    schema = request.response_format.json_schema["schema"]
                else:
                    schema = request.response_format.json_schema
                
            prompt_text = "\n".join([f"{msg.role}: {msg.content}" for msg in request.messages])
            
            structured_output = await model_router.generate_structured_output(
                prompt=prompt_text,
                schema=schema,
                max_tokens=max_tokens
            )
            
            response_text = json.dumps(structured_output)
        
        elif request.response_format and request.response_format.type == "json_object":
            system_msg = next((msg for msg in request.messages if msg.role == "system"), None)
            
            if system_msg:
                system_msg.content = f"{system_msg.content or ''}\nRespond with JSON format only."
            else:
                request.messages.insert(0, ChatMessage(
                    role="system", 
                    content="Respond with JSON format only."
                ))
            
            response_text = await model_router.generate_text(
                prompt=request.messages,
                max_tokens=max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=request.stop
            )
        else:
            response_text = await model_router.generate_text(
                prompt=request.messages,
                max_tokens=max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=request.stop
            )
        
        completion_tokens = count_tokens(response_text, request.model)
        total_tokens = prompt_tokens + completion_tokens
        
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
    start_time = time.time()
    request_id = str(uuid.uuid4())
    created_time = int(time.time())
    system_fingerprint = f"fp_{uuid.uuid4().hex[:10]}"
    
    prompt_tokens = sum(count_tokens(msg.content or "", request.model) for msg in request.messages)
    accumulated_text = ""
    
    try:
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
        
        initial_chunk = ChatCompletionChunkResponse(
            id=f"chatcmpl-{request_id}",
            created=created_time,
            model=request.model,
            system_fingerprint=system_fingerprint,
            service_tier="default", 
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=ChatCompletionChunkDelta(
                        role="assistant",
                        content="",
                        refusal=None
                    ),
                    finish_reason=None,
                    logprobs=None
                )
            ],
            object="chat.completion.chunk"
        )
        yield f"data: {initial_chunk.model_dump_json(exclude_none=False)}\n\n"
        
        if request.response_format and request.response_format.type == "json_schema":
            schema = {}
            if request.response_format.json_schema:
                if isinstance(request.response_format.json_schema, dict) and "schema" in request.response_format.json_schema:
                    schema = request.response_format.json_schema["schema"]
                else:
                    schema = request.response_format.json_schema
                
            prompt_text = "\n".join([f"{msg.role}: {msg.content}" for msg in request.messages])
            
            try:
                structured_output = await model_router.generate_structured_output(
                    prompt=prompt_text,
                    schema=schema,
                    max_tokens=max_tokens
                )
                
                response_text = json.dumps(structured_output)
                accumulated_text = response_text
                
                chunk_size = 10
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
        
        elif request.response_format and request.response_format.type == "json_object":
            system_msg = next((msg for msg in request.messages if msg.role == "system"), None)
            
            if system_msg:
                system_msg.content = f"{system_msg.content or ''}\nRespond with JSON format only."
            else:
                request.messages.insert(0, ChatMessage(
                    role="system", 
                    content="Respond with JSON format only."
                ))
            
            stream_generator = await model_router.generate_text(
                prompt=request.messages,
                max_tokens=max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=request.stop,
                stream=True
            )
            
            async for text_chunk in stream_generator:
                if text_chunk:
                    accumulated_text += text_chunk
                    
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
                if text_chunk:
                    accumulated_text += text_chunk
                    
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
        
        final_chunk = ChatCompletionChunkResponse(
            id=f"chatcmpl-{request_id}",
            created=created_time,
            model=request.model,
            system_fingerprint=system_fingerprint,
            service_tier="default",
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=ChatCompletionChunkDelta(refusal=None),
                    finish_reason="stop",
                    logprobs=None
                )
            ],
            object="chat.completion.chunk"
        )
        yield f"data: {final_chunk.model_dump_json(exclude_none=False)}\n\n"
        yield "data: [DONE]\n\n"
        
        completion_tokens = count_tokens(accumulated_text, request.model)
        total_tokens = prompt_tokens + completion_tokens
        
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
        error_chunk = {
            "error": {
                "message": str(e),
                "type": "server_error"
            }
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
        yield "data: [DONE]\n\n"

@router.post("/audio/speech", response_model=None)
async def create_audio_speech(
    request: ChatCompletionRequest,
    background_tasks: BackgroundTasks,
    speaker: str = Query("Chelsie", description="Voice to use for audio generation"),
    api_key: ApiKey = Depends(get_api_key),
    db: Session = Depends(get_db)
):
    start_time = time.time()
    request_id = str(uuid.uuid4())
    created_time = int(time.time())
    system_fingerprint = f"fp_{uuid.uuid4().hex[:10]}"
    
    try:
        prompt_tokens = sum(count_tokens(msg.content or "", request.model) for msg in request.messages)
        
        max_tokens = request.max_tokens
        model_router = await ModelRouter.initialize_from_model_name(
            model_name=request.model,
            model_type=ModelType.AUDIO_GENERATION,  # This will automatically select the Qwen Omni models
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=max_tokens,
            frequency_penalty=request.frequency_penalty,
            presence_penalty=request.presence_penalty,
            stop=request.stop
        )
        
        # Generate both text and audio
        text_output, audio_data = await model_router.generate_audio_and_text(
            prompt=request.messages,
            max_tokens=max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=request.stop,
            speaker=speaker,
            return_audio=True
        )
        
        # Calculate tokens
        completion_tokens = count_tokens(text_output, request.model)
        total_tokens = prompt_tokens + completion_tokens
        
        # Log usage
        completion_time = time.time() - start_time
        background_tasks.add_task(
            log_usage,
            db=db,
            api_key_id=getattr(api_key, "id", None),
            request_id=request_id,
            endpoint="audio/speech",
            model=request.model,
            provider=model_router.provider.value,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            processing_time=completion_time,
            request_data=request.model_dump_json()
        )
        
        # Encode audio to base64
        audio_base64 = base64.b64encode(audio_data.tobytes()).decode("utf-8")
        
        # Return response with both text and audio
        return {
            "id": f"speechgen-{request_id}",
            "created": created_time,
            "model": request.model,
            "text": text_output,
            "audio": audio_base64,
            "audio_format": "wav",
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            },
            "system_fingerprint": system_fingerprint,
            "object": "audio.speech"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/audio/speech/stream", response_model=None)
async def stream_audio_speech(
    request: ChatCompletionRequest,
    background_tasks: BackgroundTasks,
    speaker: str = Query("Chelsie", description="Voice to use for audio generation"),
    api_key: ApiKey = Depends(get_api_key),
    db: Session = Depends(get_db)
):
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    async def audio_stream_generator():
        try:
            prompt_tokens = sum(count_tokens(msg.content or "", request.model) for msg in request.messages)
            
            max_tokens = request.max_tokens
            model_router = await ModelRouter.initialize_from_model_name(
                model_name=request.model,
                model_type=ModelType.AUDIO_GENERATION,
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=max_tokens,
                frequency_penalty=request.frequency_penalty,
                presence_penalty=request.presence_penalty,
                stop=request.stop
            )
            
            # Qwen models don't support true streaming for audio yet,
            # so we'll generate the full response and then simulate streaming
            text_output, audio_data = await model_router.generate_audio_and_text(
                prompt=request.messages,
                max_tokens=max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=request.stop,
                speaker=speaker,
                return_audio=True
            )
            
            # Calculate tokens and log usage
            completion_tokens = count_tokens(text_output, request.model)
            total_tokens = prompt_tokens + completion_tokens
            
            completion_time = time.time() - start_time
            background_tasks.add_task(
                log_usage,
                db=db,
                api_key_id=getattr(api_key, "id", None),
                request_id=request_id,
                endpoint="audio/speech/stream",
                model=request.model,
                provider=model_router.provider.value,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                processing_time=completion_time,
                request_data=request.model_dump_json()
            )
            
            # First yield the text response
            text_chunk = json.dumps({
                "type": "text",
                "text": text_output,
                "id": f"speechgen-{request_id}"
            })
            yield f"data: {text_chunk}\n\n"
            
            # Then stream the audio in chunks
            if audio_data is not None:
                # Split audio into chunks (e.g., 1 second each)
                sample_rate = 24000  # Typical sample rate for Qwen Omni audio
                chunk_size = sample_rate * 1  # 1 second chunks
                audio_array = audio_data
                
                for i in range(0, len(audio_array), chunk_size):
                    chunk = audio_array[i:i + chunk_size]
                    if len(chunk) > 0:
                        # Convert chunk to bytes and base64 encode
                        chunk_bytes = chunk.tobytes()
                        chunk_base64 = base64.b64encode(chunk_bytes).decode("utf-8")
                        
                        audio_chunk = json.dumps({
                            "type": "audio_chunk",
                            "audio": chunk_base64,
                            "format": "wav",
                            "chunk_index": i // chunk_size,
                            "id": f"speechgen-{request_id}"
                        })
                        yield f"data: {audio_chunk}\n\n"
                        await asyncio.sleep(0.5)  # Small delay between chunks
            
            # Signal completion
            end_chunk = json.dumps({"type": "done", "id": f"speechgen-{request_id}"})
            yield f"data: {end_chunk}\n\n"
            
        except Exception as e:
            error_chunk = json.dumps({
                "type": "error",
                "error": {
                    "message": str(e),
                    "type": "server_error"
                }
            })
            yield f"data: {error_chunk}\n\n"
    
    return StreamingResponse(
        audio_stream_generator(),
        media_type="text/event-stream"
    )

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