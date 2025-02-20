from fastapi import FastAPI
from app.api.v1.endpoints import agent, documents, user, upload, minio_webhook

app = FastAPI()

app.include_router(agent.router, prefix="/agents", tags=["Agents"])
app.include_router(documents.router, prefix="/documents", tags=["Documents"])
app.include_router(user.router, prefix="/users", tags=["Users"])
app.include_router(upload.router, prefix="/files", tags=["File Upload"])
app.include_router(minio_webhook.router, prefix="/minio", tags=["MinIO Webhook"])