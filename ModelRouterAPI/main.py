import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
import uvicorn

# Import API routers
from api.v1.router import api_router
from db.init_db import init_db

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ModelRouter API",
    description="OpenAI-compatible API for routing requests to different model providers",
    version="0.1.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom exception handling
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": {"message": str(exc), "type": "server_error"}},
    )

# Include API routes
app.include_router(api_router, prefix="/v1")

# Custom OpenAPI schema that better matches OpenAI's format
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    
    # Customize the schema here if needed to match OpenAI's format
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Welcome to the ModelRouter API. See /docs for API documentation"}

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    logger.info("Initializing database...")
    init_db()
    logger.info("ModelRouter API server is starting up")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)