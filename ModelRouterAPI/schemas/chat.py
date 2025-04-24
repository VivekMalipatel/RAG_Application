from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Literal, Union

# --- Core Message Structures ---
class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: Optional[str] = None
    name: Optional[str] = None
    refusal: Optional[Any] = None
    annotations: Optional[List[Any]] = []

# --- Request Body ---
class ResponseFormat(BaseModel):
    type: Literal["text", "json_object", "json_schema"] = "text"
    json_schema: Optional[Dict[str, Any]] = None
    name: Optional[str] = None
    strict: Optional[bool] = None
    schema: Optional[Dict[str, Any]] = None

class StreamOptions(BaseModel):
    include_usage: Optional[bool] = None

class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    model: str
    frequency_penalty: Optional[float] = Field(default=0, ge=-2.0, le=2.0)
    logit_bias: Optional[Dict[int, float]] = None # Map of token ID to bias value
    logprobs: Optional[bool] = False
    max_tokens: Optional[int] = None # Deprecated but supported for compatibility
    max_completion_tokens: Optional[int] = None # New parameter replacing max_tokens
    n: Optional[int] = Field(default=1, ge=1) # How many choices to generate
    presence_penalty: Optional[float] = Field(default=0, ge=-2.0, le=2.0)
    response_format: Optional[ResponseFormat] = None
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None # Up to 4 sequences
    stream: Optional[bool] = False
    stream_options: Optional[StreamOptions] = None # Only used if stream=True
    temperature: Optional[float] = Field(default=1.0, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    user: Optional[str] = None # End-user identifier
    
    @validator('messages')
    def validate_messages(cls, messages):
        """Validate that there is at least one message and system message is first if present."""
        if not messages:
            raise ValueError("At least one message is required")
        
        # Ensure system message is first if present
        has_system = any(msg.role == "system" for msg in messages)
        if has_system and messages[0].role != "system":
            raise ValueError("System message must be the first message if present")
            
        return messages

# --- Response Body (Non-Streaming) ---
class ChatCompletionChoice(BaseModel):
    finish_reason: Literal["stop", "length", "content_filter"] 
    index: int
    message: ChatMessage
    logprobs: Optional[Any] = None

class TokenDetails(BaseModel):
    cached_tokens: int = 0
    audio_tokens: int = 0

class CompletionTokenDetails(BaseModel):
    reasoning_tokens: int = 0
    audio_tokens: int = 0
    accepted_prediction_tokens: int = 0
    rejected_prediction_tokens: int = 0

class UsageInfo(BaseModel):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int
    prompt_tokens_details: Optional[TokenDetails] = None
    completion_tokens_details: Optional[CompletionTokenDetails] = None

class ChatCompletionResponse(BaseModel):
    id: str = Field(..., example="chatcmpl-123")
    choices: List[ChatCompletionChoice]
    created: int = Field(..., example=1677652288) # Unix timestamp
    model: str = Field(..., example="gpt-3.5-turbo")
    system_fingerprint: Optional[str] = Field(None, example="fp_44709d6fcb")
    service_tier: Optional[str] = Field("default", example="default")
    object: Literal["chat.completion"] = "chat.completion"
    usage: UsageInfo

# --- Response Body (Streaming) ---
class ChatCompletionChunkDelta(BaseModel):
    content: Optional[str] = None
    role: Optional[Literal["system", "user", "assistant"]] = None
    refusal: Optional[Any] = None

class ChatCompletionChunkChoice(BaseModel):
    delta: ChatCompletionChunkDelta
    finish_reason: Optional[Literal["stop", "length", "content_filter"]] = None
    index: int
    logprobs: Optional[Any] = None

class ChatCompletionChunkResponse(BaseModel):
    id: str = Field(..., example="chatcmpl-123")
    choices: List[ChatCompletionChunkChoice]
    created: int = Field(..., example=1677652288) # Unix timestamp
    model: str = Field(..., example="gpt-3.5-turbo")
    system_fingerprint: Optional[str] = Field(None, example="fp_44709d6fcb")
    service_tier: Optional[str] = Field("default", example="default")
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    usage: Optional[UsageInfo] = None # Only in final chunk if stream_options.include_usage is True