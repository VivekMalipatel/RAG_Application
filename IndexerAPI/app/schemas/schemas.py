from typing import Optional, Dict, Any
from pydantic import BaseModel

class FileIngestRequest(BaseModel):
    source: str
    metadata: Optional[Dict[str, Any]] = None

class UrlIngestRequest(BaseModel):
    url: str
    source: str
    metadata: Optional[Dict[str, Any]] = None

class RawTextIngestRequest(BaseModel):
    text: str
    source: str
    metadata: Optional[Dict[str, Any]] = None

class IngestResponse(BaseModel):
    id: str
    message: str

class StatusResponse(BaseModel):
    id: str
    status: str
    message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None