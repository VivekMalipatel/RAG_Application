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

class DeleteFileRequest(BaseModel):
    user_id: str
    org_id: str
    source: str
    filename: str

class DeleteResponse(BaseModel):
    success: bool
    message: str
