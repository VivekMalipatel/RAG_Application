from fastapi import APIRouter

router = APIRouter()

# Import API endpoints
from app.api.v1.endpoints import agent, documents, user

router.include_router(agent.router, prefix="/agents", tags=["Agents"])
router.include_router(documents.router, prefix="/documents", tags=["Documents"])
router.include_router(user.router, prefix="/users", tags=["Users"])