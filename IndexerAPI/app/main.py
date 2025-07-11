from fastapi import FastAPI, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import logging
import uvicorn
import asyncio
from contextlib import asynccontextmanager

from app.config import settings
from app.api.routes import ingest, status, vector
# from app.api.routes.scraper import router as scraper_router
from app.api.routes.backup import router as backup_router
from app.db.database import init_db, get_db
from app.services.rabbitmq_consumer import RabbitMQConsumer
from app.core.model.model_handler import ModelHandler
from app.services.vector_store import VectorStore
from app.services.database_persistence import DatabasePersistence
from app.processors import register_processors
from app.queue.rabbitmq_handler import rabbitmq_handler

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

@asynccontextmanager
async def lifespan(app: FastAPI):

    db_persistence = DatabasePersistence()
    logging.info("Checking for database backup in S3...")
    await db_persistence.restore_from_s3()
    
    logging.info("Initializing database...")
    await init_db()
    logging.info("Database initialized successfully")
    
    db_session = await anext(get_db())
    model_handler = ModelHandler()
    vector_store = VectorStore()
    
    logging.info("Loading vector store from local storage or S3...")
    await vector_store._load_async()
    
    # Initialize RabbitMQ connection
    await rabbitmq_handler.connect()
    
    queue_consumer = RabbitMQConsumer(db_session, model_handler, vector_store)
    
    register_processors(queue_consumer)
    
    app.state.queue_consumer = queue_consumer
    app.state.model_handler = model_handler
    app.state.vector_store = vector_store
    app.state.db_persistence = db_persistence
    
    process_task = asyncio.create_task(queue_consumer.start_processing())
    backup_task = asyncio.create_task(db_persistence.schedule_periodic_backup(settings.AUTO_BACKUP_INTERVAL_MINUTES))
    logging.info("Queue consumer and database backup scheduler started successfully")
    
    yield
    
    if hasattr(app.state, "vector_store"):
        await app.state.vector_store.shutdown()
    
    if hasattr(app.state, "db_persistence"):
        await app.state.db_persistence.backup_to_s3()
    
    if hasattr(app.state, "queue_consumer"):
        await app.state.queue_consumer.stop_processing()
        logging.info("Queue consumer stopped")
    
    # Disconnect from RabbitMQ
    await rabbitmq_handler.disconnect()

    try:
        process_task.cancel()
        backup_task.cancel()
        await asyncio.wait_for(process_task, timeout=5.0)
        await asyncio.wait_for(backup_task, timeout=2.0)
    except asyncio.CancelledError:
        logging.info("Background tasks were cancelled during shutdown")
    except asyncio.TimeoutError:
        logging.warning("Background tasks didn't terminate within timeout")
    except Exception as e:
        logging.error(f"Error waiting for background tasks: {str(e)}")

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
app.include_router(vector.router, prefix="/vector", tags=["vector"])
# app.include_router(scraper_router, prefix="/scraper", tags=["scraper"])
app.include_router(backup_router, prefix="/backup", tags=["backup"])

@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8009, reload=True)