import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
import uvicorn
from contextlib import asynccontextmanager
import asyncio

from api.v1.router import api_router
from api.v2.router import api_router as api_router_v2
from db.init_db import init_db
from huggingface.model_cache import ModelCache

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing database...")
    init_db()
    logger.info("ModelRouter API server is starting up")

    yield
    
    logger.info("ModelRouter API server is shutting down")
    
    try:
        logger.info("Cleaning up model cache...")
        model_cache = ModelCache()
        model_cache.shutdown()
        logger.info("Model cache cleanup complete")
    except Exception as e:
        logger.error(f"Error during model cache cleanup: {e}")

app = FastAPI(
    title="ModelRouter API",
    description="OpenAI-compatible API for routing requests to different model providers",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": {"message": str(exc), "type": "server_error"}},
    )

app.include_router(api_router, prefix="/v1")
app.include_router(api_router_v2, prefix="/v2")

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Welcome to the ModelRouter API. See /docs for API documentation"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)