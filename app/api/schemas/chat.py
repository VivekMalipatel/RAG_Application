from pydantic import BaseModel, UUID4
from datetime import datetime
from typing import Optional, List, Dict

class ChatSchema(BaseModel):
    index: UUID4
    chat_id: UUID4
    user_id: UUID4
    timestamp: datetime
    message: str
    message_type: str  # "user" or "agent"
    chat_metadata: Optional[Dict] = None
    entities: Optional[List[str]] = None
    relationships: Optional[List[Dict]] = None
    chat_summary: Optional[str] = None

    class Config:
        orm_mode = True