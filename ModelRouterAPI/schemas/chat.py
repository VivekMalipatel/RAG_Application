from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Literal, Union
from config import settings

class ChatCompletionMessageToolCallFunction(BaseModel):
    arguments: str = Field(..., description="The arguments to call the function with, as generated by the model in JSON format")
    name: str = Field(..., description="The name of the function to call")

class ChatCompletionMessageToolCall(BaseModel):
    id: str = Field(..., description="The ID of the tool call")
    function: ChatCompletionMessageToolCallFunction = Field(..., description="The function that the model called")
    type: Literal["function"] = Field(..., description="The type of the tool. Currently, only function is supported")

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool", "function", "developer"]
    content: Optional[Union[str, List[dict]]] = None
    name: Optional[str] = None
    refusal: Optional[Any] = None
    tool_call_id: Optional[str] = None
    annotations: Optional[List[Any]] = None
    tool_calls: Optional[List[ChatCompletionMessageToolCall]] = None
    audio: Optional[Dict[str, Any]] = None

class ResponseFormat(BaseModel):
    type: Literal["text", "json_object", "json_schema", None]
    json_schema: Optional[Dict[str, Any]] = None
    name: Optional[str] = None
    strict: Optional[bool] = None
    schema: Optional[Dict[str, Any]] = None

class StreamOptions(BaseModel):
    include_usage: Optional[bool] = None

class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    model: str
    frequency_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)
    logit_bias: Optional[Dict[str, float]] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = Field(default=None, ge=0, le=20)
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    n: Optional[int] = Field(default=None, ge=1)
    presence_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)
    response_format: Optional[ResponseFormat] = None
    seed: Optional[int] = None
    service_tier: Optional[str] = None
    stop: Optional[Union[str, List[str]]] = None
    store: Optional[bool] = None
    stream: Optional[bool] = None
    stream_options: Optional[StreamOptions] = None
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    parallel_tool_calls: Optional[bool] = None
    user: Optional[str] = None
    metadata: Optional[Dict[str, str]] = None
    reasoning_effort: Optional[str] = None
    modalities: Optional[List[str]] = None
    audio: Optional[Dict[str, Any]] = None
    prediction: Optional[Dict[str, Any]] = None
    web_search_options: Optional[Dict[str, Any]] = None
    
    num_ctx: Optional[int] = Field(default=None, description="Context window size (Ollama)")
    repeat_last_n: Optional[int] = Field(default=None, description="How far back to look for repetition (Ollama)")
    repeat_penalty: Optional[float] = Field(default=None, description="Repetition penalty (Ollama)")
    top_k: Optional[int] = Field(default=None, description="Top-k sampling (Ollama)")
    min_p: Optional[float] = Field(default=None, description="Min-p sampling (Ollama)")
    keep_alive: Optional[str] = Field(default=None, description="Keep model alive duration (Ollama)")
    think: Optional[bool] = Field(default=None, description="Enable thinking mode (Ollama)")
    
    @validator('messages')
    def validate_messages(cls, messages):
        if not messages:
            raise ValueError("At least one message is required")
        
        has_system = any(msg.role == "system" for msg in messages)
        if has_system and messages[0].role != "system":
            raise ValueError("System message must be the first message if present")
            
        return messages

class ChatCompletionChoice(BaseModel):
    finish_reason: Literal["stop", "length", "content_filter", "tool_calls"] 
    index: int
    message: ChatMessage
    logprobs: Optional[Any] = None

class TokenDetails(BaseModel):
    cached_tokens: int = None
    audio_tokens: int = None

class CompletionTokenDetails(BaseModel):
    reasoning_tokens: int = None
    audio_tokens: int = None
    accepted_prediction_tokens: int = None
    rejected_prediction_tokens: int = None

class UsageInfo(BaseModel):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int
    prompt_tokens_details: Optional[TokenDetails] = None
    completion_tokens_details: Optional[CompletionTokenDetails] = None

class ChatCompletionResponse(BaseModel):
    id: str = Field(..., example="chatcmpl-123")
    choices: List[ChatCompletionChoice]
    created: int = Field(..., example=1677652288)
    model: str = Field(..., example="gpt-3.5-turbo")
    system_fingerprint: Optional[str] = Field(None, example="fp_44709d6fcb")
    service_tier: Optional[str] = Field(None, example="default")
    object: Literal["chat.completion"] = None
    usage: UsageInfo

class ChatCompletionChunkDelta(BaseModel):
    content: Optional[str] = None
    role: Optional[Literal["system", "user", "assistant", "tool", "function", "developer"]] = None
    refusal: Optional[Any] = None
    tool_calls: Optional[List[ChatCompletionMessageToolCall]] = None

class ChatCompletionChunkChoice(BaseModel):
    delta: ChatCompletionChunkDelta
    finish_reason: Optional[Literal["stop", "length", "content_filter", "tool_calls"]] = None
    index: int
    logprobs: Optional[Any] = None

class ChatCompletionChunkResponse(BaseModel):
    id: str = Field(..., example="chatcmpl-123")
    choices: List[ChatCompletionChunkChoice]
    created: int = Field(..., example=1677652288)
    model: str = Field(..., example="gpt-3.5-turbo")
    system_fingerprint: Optional[str] = Field(None, example="fp_44709d6fcb")
    service_tier: Optional[str] = Field(None, example="default")
    object: Literal["chat.completion.chunk"] = None
    usage: Optional[UsageInfo] = None