from fastapi import FastAPI
from app.api.v1 import api_router
from app.api.core.config import settings

app = FastAPI(title="OmniRAG API", version="1.0")

app.include_router(api_router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"message": "OmniRAG API is running!"}