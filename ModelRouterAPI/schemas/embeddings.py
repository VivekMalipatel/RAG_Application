from pydantic import BaseModel, Field
from typing import List, Optional, Union, Dict, Any

from schemas.chat import UsageInfo

class EmbeddingRequest(BaseModel):
    model: str = Field(..., description="ID of the model to use")
    input: Union[str, List[str], List[dict]] = Field(..., description="Input text to embed, encoded as a string or array of strings")
    user: Optional[str] = Field(None, description="A unique identifier representing your end-user")
    encoding_format: Optional[str] = Field("float", description="The format to return the embeddings in. Can be either float or base64")
    dimensions: Optional[Union[int, Any]] = Field(None, description="The number of dimensions the resulting output embeddings should have")
    
    # Hidden field for extra parameters not in the OpenAI API but useful for our implementation
    model_extra: Dict[str, Any] = Field(default_factory=dict, exclude=True)

class EmbeddingData(BaseModel):
    object: str = "embedding"
    # Allow for both flat embeddings and nested embeddings from Nomic multimodal models
    embedding: Union[List[float], List[List[float]]]
    index: int

class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: UsageInfo