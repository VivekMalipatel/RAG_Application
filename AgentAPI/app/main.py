from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from api.routes import chat, models, tools
from db.redis import redis

@asynccontextmanager
async def lifespan(app: FastAPI):
    redis.init_session()
    yield
    redis.close_session()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(chat.router)
app.include_router(models.router)
app.include_router(tools.router)

def root():
    return {"status": "ok"}

app.add_api_route("/", root, methods=["GET"])
