from fastapi import APIRouter

router = APIRouter()

@router.get("/")
async def health_check():
    return {"status": "healthy", "service": "Agent API"}

@router.get("/detailed")
async def detailed_health_check():
    return {
        "status": "healthy",
        "service": "Agent API",
        "version": "1.0.0",
        "components": {
            "agents": "healthy",
            "memory": "healthy",
            "llm": "healthy"
        }
    }
