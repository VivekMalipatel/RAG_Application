from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Union, Literal, Any

class ChatMessage(BaseModel):
    role: str
    content: str
    name: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    
    @validator('messages')
    def validate_messages(cls, messages):
        if not messages:
            raise ValueError("At least one message is required")
        
        # Check that system message comes first if present
        if messages and messages[0].role != "system" and any(m.role == "system" for m in messages[1:]):
            raise ValueError("System message must be the first message")
        
        return messages

class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None

class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int  # Unix timestamp
    model: str
    choices: List[ChatCompletionChoice]
    usage: UsageInfo


# For streaming responses
class ChatCompletionChunkDelta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None

class ChatCompletionChunkChoice(BaseModel):
    index: int
    delta: ChatCompletionChunkDelta
    finish_reason: Optional[str] = None

class ChatCompletionChunkResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionChunkChoice]