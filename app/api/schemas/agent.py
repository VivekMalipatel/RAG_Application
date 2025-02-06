from pydantic import BaseModel
from datetime import datetime

class AgentBase(BaseModel):
    name: str
    description: str | None = None
    endpoint: str
    is_active: bool = True

class AgentCreate(AgentBase):
    pass

class AgentUpdate(AgentBase):
    pass

class AgentResponse(AgentBase):
    id: int
    created_at: datetime
    updated_at: datetime | None

    class Config:
        from_attributes = True

class DeleteAgentResponse(BaseModel):
    message: str
    name: str