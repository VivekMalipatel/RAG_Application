from fastapi import FastAPI, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import logging
import uvicorn
import asyncio
from contextlib import asynccontextmanager

from config import settings
from api.routes import ingest, status, vector
from api.routes.backup import router as backup_router
from core.db.database import init_db, cleanup_global_db_session
from core.model.model_handler import get_global_model_handler
from services.vector_store import get_global_vector_store, cleanup_global_vector_store
from services.database_persistence import get_global_db_persistence
from services.rabbitmq_consumer import get_global_rabbitmq_consumer, cleanup_global_rabbitmq_consumer
from core.processors import register_processors
from core.queue.rabbitmq_handler import rabbitmq_handler

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

@asynccontextmanager
async def lifespan(app: FastAPI):

    db_persistence = get_global_db_persistence()
    logging.info("Checking for database backup in S3...")
    await db_persistence.restore_from_s3()
    
    logging.info("Initializing database...")
    await init_db()
    logging.info("Database initialized successfully")
    
    model_handler = get_global_model_handler()
    vector_store = get_global_vector_store()
    
    logging.info("Loading vector store from local storage or S3...")
    await vector_store._load_async()
    
    await rabbitmq_handler.connect()
    
    rabbitmq_consumer = get_global_rabbitmq_consumer()
    
    register_processors(rabbitmq_consumer)
    
    process_task = asyncio.create_task(rabbitmq_consumer.start_processing())
    backup_task = asyncio.create_task(db_persistence.schedule_periodic_backup(settings.AUTO_BACKUP_INTERVAL_MINUTES))
    logging.info("RabbitMQ consumer and database backup scheduler started successfully")
    
    yield
    
    await cleanup_global_vector_store()
    
    await db_persistence.backup_to_s3()
    
    await cleanup_global_rabbitmq_consumer()
    logging.info("RabbitMQ consumer stopped")
    
    await rabbitmq_handler.disconnect()

    await cleanup_global_db_session()

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
app.include_router(backup_router, prefix="/backup", tags=["backup"])

@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8009, reload=True)