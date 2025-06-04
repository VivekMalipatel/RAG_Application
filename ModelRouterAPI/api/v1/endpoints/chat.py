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
    ChatCompletionRequest, ChatCompletionResponse, ChatCompletionChoice, ChatMessage, ResponseFormat,
    ChatCompletionChunkResponse, ChatCompletionChunkChoice, ChatCompletionChunkDelta, UsageInfo,
    ChatCompletionMessageToolCall, ChatCompletionMessageToolCallFunction
)

from model_handler import ModelRouter
from model_type import ModelType
from config import settings

router = APIRouter()

def count_tokens(text: str, model: str = "gpt-4o") -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except KeyError:
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception:
            return len(text.split()) * 1.3

def apply_chat_defaults(request: ChatCompletionRequest) -> None:
    if request.frequency_penalty is None:
        request.frequency_penalty = settings.CHAT_DEFAULT_FREQUENCY_PENALTY
    # if request.logprobs is None:
    #     request.logprobs = settings.CHAT_DEFAULT_LOGPROBS
    # if request.n is None:
    #     request.n = settings.CHAT_DEFAULT_N
    # if request.presence_penalty is None:
    #     request.presence_penalty = settings.CHAT_DEFAULT_PRESENCE_PENALTY
    if request.stream is None:
        request.stream = settings.CHAT_DEFAULT_STREAM
    if request.temperature is None:
        request.temperature = settings.CHAT_DEFAULT_TEMPERATURE
    if request.top_p is None:
        request.top_p = settings.CHAT_DEFAULT_TOP_P
    # if request.parallel_tool_calls is None:
    #     request.parallel_tool_calls = settings.CHAT_DEFAULT_PARALLEL_TOOL_CALLS
    # if request.response_format is None:
    #     request.response_format = ResponseFormat(type=settings.CHAT_DEFAULT_RESPONSE_FORMAT_TYPE)
    # elif request.response_format.type is None:
    #     request.response_format.type = settings.CHAT_DEFAULT_RESPONSE_FORMAT_TYPE
    # if request.service_tier is None:
    #     request.service_tier = settings.CHAT_DEFAULT_SERVICE_TIER
    
    for message in request.messages:
        if message.annotations is None:
            message.annotations = []

@router.post("/chat/completions", response_model=None)
async def create_chat_completion(
    request: ChatCompletionRequest,
    background_tasks: BackgroundTasks,
    api_key: ApiKey = Depends(get_api_key),
    db: Session = Depends(get_db)
):
    apply_chat_defaults(request)
    
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
        
        try:
            model_router = await ModelRouter.initialize_from_model_name(
                model_name=request.model,
                model_type=ModelType.TEXT_GENERATION,
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=max_tokens,
                max_completion_tokens=request.max_completion_tokens,
                frequency_penalty=request.frequency_penalty,
                presence_penalty=request.presence_penalty,
                stop=request.stop,
                logit_bias=request.logit_bias,
                logprobs=request.logprobs,
                top_logprobs=request.top_logprobs,
                n=request.n,
                seed=request.seed,
                user=request.user,
                tools=request.tools,
                tool_choice=request.tool_choice,
                parallel_tool_calls=request.parallel_tool_calls,
                response_format=request.response_format.model_dump() if request.response_format else None,
                service_tier=request.service_tier,
                store=request.store,
                metadata=request.metadata,
                reasoning_effort=request.reasoning_effort,
                modalities=request.modalities,
                audio=request.audio,
                prediction=request.prediction,
                web_search_options=request.web_search_options,
                stream_options=request.stream_options.model_dump() if request.stream_options else None,
                num_ctx=request.num_ctx,
                repeat_last_n=request.repeat_last_n,
                repeat_penalty=request.repeat_penalty,
                top_k=request.top_k,
                min_p=request.min_p,
                keep_alive=request.keep_alive,
                think=request.think
            )
        except Exception as model_init_error:
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to initialize model router for model '{request.model}': {str(model_init_error)}"
            )
        
        response_text = ""
        tool_calls = None
        response_message = None
        
        if request.response_format and request.response_format.type == "json_schema":
            schema = {}
            if request.response_format.json_schema:
                if isinstance(request.response_format.json_schema, dict) and "schema" in request.response_format.json_schema:
                    schema = request.response_format.json_schema["schema"]
                else:
                    schema = request.response_format.json_schema
                
            messages_for_api = []
            for msg in request.messages:
                msg_dict = {"role": msg.role}
                if msg.content:
                    msg_dict["content"] = msg.content
                if msg.name:
                    msg_dict["name"] = msg.name
                messages_for_api.append(msg_dict)
            
            try:
                structured_output = await model_router.generate_structured_output(
                    prompt=messages_for_api,
                    schema=schema,
                    max_tokens=max_tokens,
                    max_completion_tokens=request.max_completion_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    stop=request.stop,
                    logit_bias=request.logit_bias,
                    logprobs=request.logprobs,
                    top_logprobs=request.top_logprobs,
                    n=request.n,
                    seed=request.seed,
                    user=request.user,
                    tools=request.tools,
                    tool_choice=request.tool_choice,
                    parallel_tool_calls=request.parallel_tool_calls,
                    service_tier=request.service_tier,
                    store=request.store,
                    metadata=request.metadata,
                    reasoning_effort=request.reasoning_effort,
                    modalities=request.modalities,
                    audio=request.audio,
                    prediction=request.prediction,
                    web_search_options=request.web_search_options,
                    num_ctx=request.num_ctx,
                    repeat_last_n=request.repeat_last_n,
                    repeat_penalty=request.repeat_penalty,
                    top_k=request.top_k,
                    min_p=request.min_p,
                    keep_alive=request.keep_alive,
                    think=request.think
                )
                response_text = json.dumps(structured_output)
            except Exception as structured_error:
                raise HTTPException(
                    status_code=500, 
                    detail=f"Failed to generate structured output: {str(structured_error)}"
                )

        elif request.response_format and request.response_format.type == "json_object":
            system_msg = next((msg for msg in request.messages if msg.role == "system"), None)
            
            if system_msg:
                system_msg.content = f"{system_msg.content or ''}\nRespond with JSON format only."
            else:
                request.messages.insert(0, ChatMessage(
                    role="system", 
                    content="Respond with JSON format only."
                ))
            
            try:
                response_text = await model_router.generate_text(
                    prompt=request.messages,
                    max_tokens=max_tokens,
                    max_completion_tokens=request.max_completion_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    stop=request.stop,
                    logit_bias=request.logit_bias,
                    logprobs=request.logprobs,
                    top_logprobs=request.top_logprobs,
                    n=request.n,
                    seed=request.seed,
                    user=request.user,
                    tools=request.tools,
                    tool_choice=request.tool_choice,
                    parallel_tool_calls=request.parallel_tool_calls,
                    service_tier=request.service_tier,
                    store=request.store,
                    metadata=request.metadata,
                    reasoning_effort=request.reasoning_effort,
                    modalities=request.modalities,
                    audio=request.audio,
                    prediction=request.prediction,
                    web_search_options=request.web_search_options,
                    num_ctx=request.num_ctx,
                    repeat_last_n=request.repeat_last_n,
                    repeat_penalty=request.repeat_penalty,
                    top_k=request.top_k,
                    min_p=request.min_p,
                    keep_alive=request.keep_alive,
                    think=request.think
                )
            except Exception as json_generation_error:
                raise HTTPException(
                    status_code=500, 
                    detail=f"Failed to generate JSON object response: {str(json_generation_error)}"
                )
        else:
            try:
                response_message = await model_router.generate_text(
                    prompt=request.messages,
                    max_tokens=max_tokens,
                    max_completion_tokens=request.max_completion_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    stop=request.stop,
                    logit_bias=request.logit_bias,
                    logprobs=request.logprobs,
                    top_logprobs=request.top_logprobs,
                    n=request.n,
                    seed=request.seed,
                    user=request.user,
                    tools=request.tools,
                    tool_choice=request.tool_choice,
                    parallel_tool_calls=request.parallel_tool_calls,
                    service_tier=request.service_tier,
                    store=request.store,
                    metadata=request.metadata,
                    reasoning_effort=request.reasoning_effort,
                    modalities=request.modalities,
                    audio=request.audio,
                    prediction=request.prediction,
                    web_search_options=request.web_search_options,
                    num_ctx=request.num_ctx,
                    repeat_last_n=request.repeat_last_n,
                    repeat_penalty=request.repeat_penalty,
                    top_k=request.top_k,
                    min_p=request.min_p,
                    keep_alive=request.keep_alive,
                    think=request.think
                )
                
                if hasattr(response_message, 'content'):
                    response_text = response_message.content or ""
                    if hasattr(response_message, 'tool_calls') and response_message.tool_calls:
                        tool_calls = []
                        for tool_call in response_message.tool_calls:
                            tool_calls.append(ChatCompletionMessageToolCall(
                                id=tool_call.id,
                                type=tool_call.type,
                                function=ChatCompletionMessageToolCallFunction(
                                    name=tool_call.function.name,
                                    arguments=tool_call.function.arguments
                                )
                            ))
                elif isinstance(response_message, str):
                    response_text = response_message
                    tool_calls = None
                else:
                    response_text = str(response_message)
                    tool_calls = None
                    
            except Exception as text_generation_error:
                raise HTTPException(
                    status_code=500, 
                    detail=f"Failed to generate text response: {str(text_generation_error)}"
                )
        
        if isinstance(response_message, list):
            choices = []
            total_completion_tokens = 0
            
            for i, item in enumerate(response_message):
                if hasattr(item, 'content'):
                    content = item.content
                    item_tool_calls = None
                    if hasattr(item, 'tool_calls') and item.tool_calls:
                        item_tool_calls = []
                        for tool_call in item.tool_calls:
                            item_tool_calls.append(ChatCompletionMessageToolCall(
                                id=tool_call.id,
                                type=tool_call.type,
                                function=ChatCompletionMessageToolCallFunction(
                                    name=tool_call.function.name,
                                    arguments=tool_call.function.arguments
                                )
                            ))
                else:
                    content = str(item)
                    item_tool_calls = None
                
                completion_tokens_single = count_tokens(content or "", request.model)
                total_completion_tokens += completion_tokens_single
                
                choices.append(
                    ChatCompletionChoice(
                        index=i,
                        message=ChatMessage(
                            role="assistant", 
                            content=content if content else None,
                            refusal=None,
                            annotations=[],
                            tool_calls=item_tool_calls
                        ),
                        finish_reason="tool_calls" if item_tool_calls else "stop",
                        logprobs=None
                    )
                )
            completion_tokens = total_completion_tokens
        elif response_message and (hasattr(response_message, 'content') or hasattr(response_message, 'tool_calls')):
            content = getattr(response_message, 'content', None)
            message_tool_calls = None
            
            if hasattr(response_message, 'tool_calls') and response_message.tool_calls:
                message_tool_calls = []
                for tool_call in response_message.tool_calls:
                    message_tool_calls.append(ChatCompletionMessageToolCall(
                        id=tool_call.id,
                        type=tool_call.type,
                        function=ChatCompletionMessageToolCallFunction(
                            name=tool_call.function.name,
                            arguments=tool_call.function.arguments
                        )
                    ))
            
            completion_tokens = count_tokens(content or "", request.model)
            choices = [
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content=content,
                        refusal=None,
                        annotations=[],
                        tool_calls=message_tool_calls
                    ),
                    finish_reason="tool_calls" if message_tool_calls else "stop",
                    logprobs=None
                )
            ]
        else:
            completion_tokens = count_tokens(response_text or "", request.model)
            choices = [
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant", 
                        content=response_text if response_text else None,
                        refusal=None,
                        annotations=[],
                        tool_calls=tool_calls
                    ),
                    finish_reason="tool_calls" if tool_calls else "stop",
                    logprobs=None
                )
            ]
            
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
            choices=choices,
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
            service_tier=request.service_tier or settings.CHAT_DEFAULT_SERVICE_TIER,
            object=settings.CHAT_DEFAULT_OBJECT_COMPLETION
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error in chat completion: {str(e)}"
        )

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
        
        try:
            model_router = await ModelRouter.initialize_from_model_name(
                model_name=request.model,
                model_type=ModelType.TEXT_GENERATION,
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=max_tokens,
                max_completion_tokens=request.max_completion_tokens,
                frequency_penalty=request.frequency_penalty,
                presence_penalty=request.presence_penalty,
                stop=request.stop,
                logit_bias=request.logit_bias,
                logprobs=request.logprobs,
                top_logprobs=request.top_logprobs,
                n=request.n,
                seed=request.seed,
                user=request.user,
                tools=request.tools,
                tool_choice=request.tool_choice,
                parallel_tool_calls=request.parallel_tool_calls,
                response_format=request.response_format.model_dump() if request.response_format else None,
                service_tier=request.service_tier,
                store=request.store,
                metadata=request.metadata,
                reasoning_effort=request.reasoning_effort,
                modalities=request.modalities,
                audio=request.audio,
                prediction=request.prediction,
                web_search_options=request.web_search_options,
                stream_options=request.stream_options.model_dump() if request.stream_options else None,
                num_ctx=request.num_ctx,
                repeat_last_n=request.repeat_last_n,
                repeat_penalty=request.repeat_penalty,
                top_k=request.top_k,
                min_p=request.min_p,
                keep_alive=request.keep_alive,
                think=request.think,
                stream=True
            )
        except Exception as model_init_error:
            error_chunk = {
                "error": {
                    "message": f"Failed to initialize model router for streaming with model '{request.model}': {str(model_init_error)}",
                    "type": "model_initialization_error"
                }
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
            yield "data: [DONE]\n\n"
            return
        
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
            except Exception as structured_stream_error:
                error_chunk = {
                    "error": {
                        "message": f"Failed to generate structured output for streaming: {str(structured_stream_error)}",
                        "type": "structured_output_error"
                    }
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"
                yield "data: [DONE]\n\n"
                return
        
        elif request.response_format and request.response_format.type == "json_object":
            system_msg = next((msg for msg in request.messages if msg.role == "system"), None)
            
            if system_msg:
                system_msg.content = f"{system_msg.content or ''}\nRespond with JSON format only."
            else:
                request.messages.insert(0, ChatMessage(
                    role="system", 
                    content="Respond with JSON format only."
                ))
            
            try:
                stream_generator = await model_router.generate_text(
                    prompt=request.messages,
                    max_tokens=max_tokens,
                    max_completion_tokens=request.max_completion_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    stop=request.stop,
                    stream=True,
                    logit_bias=request.logit_bias,
                    logprobs=request.logprobs,
                    top_logprobs=request.top_logprobs,
                    n=request.n,
                    seed=request.seed,
                    user=request.user,
                    tools=request.tools,
                    tool_choice=request.tool_choice,
                    parallel_tool_calls=request.parallel_tool_calls,
                    service_tier=request.service_tier,
                    store=request.store,
                    metadata=request.metadata,
                    reasoning_effort=request.reasoning_effort,
                    modalities=request.modalities,
                    audio=request.audio,
                    prediction=request.prediction,
                    web_search_options=request.web_search_options,
                    stream_options=request.stream_options.model_dump() if request.stream_options else None,
                    num_ctx=request.num_ctx,
                    repeat_last_n=request.repeat_last_n,
                    repeat_penalty=request.repeat_penalty,
                    top_k=request.top_k,
                    min_p=request.min_p,
                    keep_alive=request.keep_alive,
                    think=request.think
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
            except Exception as json_stream_error:
                error_chunk = {
                    "error": {
                        "message": f"Failed to generate JSON object streaming response: {str(json_stream_error)}",
                        "type": "json_streaming_error"
                    }
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"
                yield "data: [DONE]\n\n"
                return
        
        else:
            stream_generator = await model_router.generate_text(
                prompt=request.messages,
                max_tokens=max_tokens,
                max_completion_tokens=request.max_completion_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=request.stop,
                stream=True,
                logit_bias=request.logit_bias,
                logprobs=request.logprobs,
                top_logprobs=request.top_logprobs,
                n=request.n,
                seed=request.seed,
                user=request.user,
                tools=request.tools,
                tool_choice=request.tool_choice,
                parallel_tool_calls=request.parallel_tool_calls,
                service_tier=request.service_tier,
                store=request.store,
                metadata=request.metadata,
                reasoning_effort=request.reasoning_effort,
                modalities=request.modalities,
                audio=request.audio,
                prediction=request.prediction,
                web_search_options=request.web_search_options,
                stream_options=request.stream_options.model_dump() if request.stream_options else None,
                num_ctx=request.num_ctx,
                repeat_last_n=request.repeat_last_n,
                repeat_penalty=request.repeat_penalty,
                top_k=request.top_k,
                min_p=request.min_p,
                keep_alive=request.keep_alive,
                think=request.think
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