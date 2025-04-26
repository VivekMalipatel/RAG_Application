from typing import Dict, Optional, Any, List
from pydantic import BaseModel, Field, HttpUrl
from datetime import datetime
from enum import Enum

class StatusEnum(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class FileIngestRequest(BaseModel):
    source: str = Field(..., description="Source of the file")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class UrlIngestRequest(BaseModel):
    url: HttpUrl = Field(..., description="URL to ingest")
    source: str = Field(..., description="Source of the URL")
    space_id: Optional[str] = Field(None, description="Space ID for organization")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class RawTextIngestRequest(BaseModel):
    text: str = Field(..., description="Raw text to ingest")
    source: str = Field(..., description="Source of the text")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class IngestResponse(BaseModel):
    id: str = Field(..., description="Unique identifier for tracking the submission")
    message: str = Field(..., description="Status message")

class StatusResponse(BaseModel):
    id: str = Field(..., description="Unique identifier")
    status: StatusEnum = Field(..., description="Current processing status")
    message: Optional[str] = Field(None, description="Status message")
    result: Optional[Dict[str, Any]] = Field(None, description="Processing result if available")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")