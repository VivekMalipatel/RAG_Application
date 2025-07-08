"""
Main FastAPI app entrypoint. Registers all routers.
"""
from fastapi import FastAPI
from app.endpoints.routes import agents

app = FastAPI()

# Register routers
app.include_router(agents.router)

# Optionally, add a root endpoint or health check
def root():
    return {"status": "ok"}

app.add_api_route("/", root, methods=["GET"])
