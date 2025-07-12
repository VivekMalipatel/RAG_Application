from fastapi import FastAPI
from contextlib import asynccontextmanager
from api.routes import chat, models
from db.redis import redis

@asynccontextmanager
async def lifespan(app: FastAPI):
    redis.init_session()
    yield
    redis.close_session()

app = FastAPI(lifespan=lifespan)

app.include_router(chat.router)
app.include_router(models.router)

def root():
    return {"status": "ok"}

app.add_api_route("/", root, methods=["GET"])
