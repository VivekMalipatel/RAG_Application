import logging
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
import uvicorn
from contextlib import asynccontextmanager
from api.v1.router import api_router
from config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

async def api_key_middleware(request: Request, call_next):
    if request.url.path in ["/", "/docs", "/redoc", "/openapi.json"]:
        response = await call_next(request)
        return response
    
    api_key = None
    
    if settings.USE_BEARER_TOKEN:
        auth_header = request.headers.get(settings.BEARER_TOKEN_HEADER)
        if auth_header and auth_header.startswith("Bearer "):
            api_key = auth_header[7:]
    
    if not api_key:
        api_key = request.headers.get(settings.API_KEY_HEADER)
    
    if not api_key:
        return JSONResponse(
            status_code=401,
            content={"error": {"message": "Missing API key", "type": "authentication_error"}},
        )
    
    if api_key not in settings.API_KEYS:
        return JSONResponse(
            status_code=401,
            content={"error": {"message": "Invalid API key", "type": "authentication_error"}},
        )
    
    response = await call_next(request)
    return response

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("ModelRouter API server is starting up")

    yield
    
    logger.info("ModelRouter API server is shutting down")

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

@app.middleware("http")
async def api_key_check_middleware(request: Request, call_next):
    return await api_key_middleware(request, call_next)

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": {"message": str(exc), "type": "server_error"}},
    )

app.include_router(api_router, prefix="/v1")

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