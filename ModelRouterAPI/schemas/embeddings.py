from pydantic import BaseModel, Field
from typing import List, Optional, Union

from schemas.chat import UsageInfo

class EmbeddingRequest(BaseModel):
    model: str
    input: Union[str, List[str]]
    user: Optional[str] = None
    encoding_format: Optional[str] = "float"  # Can be "float" or "base64"

class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int

class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: UsageInfo