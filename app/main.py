from fastapi import FastAPI
from app.api.v1 import api_router
from app.api.db.base import Base, engine

Base.metadata.create_all(bind=engine)

app = FastAPI(title="OmniRAG Assistant")
app.include_router(api_router, prefix="/api/v1")