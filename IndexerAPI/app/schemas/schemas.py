from typing import Optional, Dict, Any
from pydantic import BaseModel

class FileIngestRequest(BaseModel):
    user_id: str
    org_id: str
    s3_url: str
    source: str
    metadata: dict

class UrlIngestRequest(BaseModel):
    user_id: str
    org_id: str
    url: str
    source: str
    metadata: dict

class RawTextIngestRequest(BaseModel):
    user_id: str
    org_id: str
    text: str
    source: str
    metadata: dict

class IngestResponse(BaseModel):
    task_id: str