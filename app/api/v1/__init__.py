from fastapi import APIRouter
from .endpoints import router as api_router

router = APIRouter()
router.include_router(api_router)