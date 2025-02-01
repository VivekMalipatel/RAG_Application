from fastapi import FastAPI
from app.api.v1.endpoints import search  # Import your API

app = FastAPI(title="OmniRAG Assistant")

# Include API endpoints
app.include_router(search.router, prefix="/search", tags=["Search"])

@app.get("/")
def root():
    return {"message": "🚀 OmniRAG API is running!"}