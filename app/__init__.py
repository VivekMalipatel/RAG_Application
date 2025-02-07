from fastapi import FastAPI

app = FastAPI()

from app.api.v1.endpoints import agent, documents, user

app.include_router(agent.router, prefix="/agents", tags=["Agents"])
app.include_router(documents.router, prefix="/documents", tags=["Documents"])
app.include_router(user.router, prefix="/users", tags=["Users"])