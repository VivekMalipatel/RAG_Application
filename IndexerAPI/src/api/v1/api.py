from fastapi import APIRouter
from api.v1.endpoints import ingest, delete, search

api_router = APIRouter()
api_router.include_router(ingest.router, tags=["ingest"], prefix="/ingest")
api_router.include_router(delete.router, tags=["delete"], prefix="/delete")
api_router.include_router(search.router, tags=["search"], prefix="/search")
