from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union, Any

from schemas.chat import UsageInfo

class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]]
    suffix: Optional[str] = None
    max_tokens: Optional[int] = 16
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    logprobs: Optional[int] = None
    echo: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    best_of: Optional[int] = 1
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None

class CompletionChoice(BaseModel):
    text: str
    index: int
    logprobs: Optional[Any] = None
    finish_reason: Optional[str] = None

class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: UsageInfo

# For streaming responses
class CompletionChunkChoice(BaseModel):
    text: str
    index: int
    logprobs: Optional[Any] = None
    finish_reason: Optional[str] = None

class CompletionChunkResponse(BaseModel):
    id: str
    object: str = "text_completion.chunk"
    created: int
    model: str
    choices: List[CompletionChunkChoice]