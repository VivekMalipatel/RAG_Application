from fastapi import FastAPI, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import logging
import uvicorn
import asyncio
from contextlib import asynccontextmanager

from app.config import settings
from app.api.routes import ingest, status
from app.db.database import init_db, get_db
from app.services.queue_consumer import QueueConsumer
from app.core.model.model_handler import ModelHandler
from app.processors import register_processors

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

@asynccontextmanager
async def lifespan(app: FastAPI):

    logging.info("Initializing database...")
    await init_db()
    logging.info("Database initialized successfully")
    
    db_session = await anext(get_db())
    model_handler = ModelHandler()
    queue_consumer = QueueConsumer(db_session, model_handler)
    
    register_processors(queue_consumer)
    
    app.state.queue_consumer = queue_consumer
    
    process_task = asyncio.create_task(queue_consumer.start_processing())
    logging.info("Queue consumer started successfully")
    
    yield
    
    if hasattr(app.state, "queue_consumer"):
        app.state.queue_consumer.stop_processing()
        logging.info("Queue consumer stopped")

    try:
        process_task.cancel()
        await asyncio.wait_for(process_task, timeout=5.0)
    except asyncio.TimeoutError:
        logging.warning("Queue processing task didn't terminate within timeout")
    except Exception as e:
        logging.error(f"Error waiting for queue processing task: {str(e)}")

app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest.router, prefix="/ingest", tags=["ingest"])
app.include_router(status.router, tags=["status"])

@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8009, reload=True)