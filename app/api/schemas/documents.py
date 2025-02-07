from pydantic import BaseModel
from datetime import datetime
from enum import Enum

class QuadrantStatus(str, Enum):
    Q1 = "Q1"
    Q2 = "Q2"
    Q3 = "Q3"
    Q4 = "Q4"
    UNASSIGNED = "UNASSIGNED"

class DocumentBase(BaseModel):
    file_url: str
    file_type: str
    description: str | None = None
    quadrant_status: QuadrantStatus = QuadrantStatus.UNASSIGNED

class DocumentCreate(DocumentBase):
    user_id: int

class DocumentUpdate(BaseModel):
    description: str | None = None
    quadrant_status: QuadrantStatus | None = None

class DocumentResponse(DocumentBase):
    id: int
    user_id: int
    created_at: datetime
    updated_at: datetime | None

    class Config:
        from_attributes = True

