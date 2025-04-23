from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class ModelPermission(BaseModel):
    id: str
    object: str = "model_permission"
    created: int
    allow_create_engine: bool
    allow_sampling: bool
    allow_logprobs: bool
    allow_search_indices: bool
    allow_view: bool
    allow_fine_tuning: bool
    organization: str
    group: Optional[str] = None
    is_blocking: bool

class Model(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: List[ModelPermission] = []

class ModelList(BaseModel):
    object: str = "list"
    data: List[Model]