import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
BACKEND_LOCAL_DIR = ROOT_DIR / "backend"
BACKEND_CONTAINER_DIR = ROOT_DIR.parent / "outer-backend"

for candidate in (ROOT_DIR, BASE_DIR):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

for candidate in (BACKEND_LOCAL_DIR, BACKEND_CONTAINER_DIR):
    if candidate.is_dir():
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import chat, copilotkit, models, tools
from backend.routes import admin as admin_routes
from backend.routes import agents as agent_routes
from backend.routes import auth as auth_routes
from backend.db import init_engine, shutdown_engine
from app.db.redis import redis

@asynccontextmanager
async def lifespan(app: FastAPI):
    redis.init_session()
    await init_engine()
    try:
        yield
    finally:
        redis.close_session()
        await shutdown_engine()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router)
app.include_router(copilotkit.router)
app.include_router(models.router)
app.include_router(tools.router)
app.include_router(auth_routes.router)
app.include_router(admin_routes.router)
app.include_router(agent_routes.router)

def root():
    return {"status": "ok"}

app.add_api_route("/", root, methods=["GET"])
