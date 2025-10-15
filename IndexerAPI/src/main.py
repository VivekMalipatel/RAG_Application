import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from core.config import settings
from api import api_router
from services.orchestrator import get_global_orchestrator, cleanup_global_orchestrator
from core.processors import register_processors
from core.queue.rabbitmq_handler import rabbitmq_handler
from core.rmq_client import ensure_rmq_connection, close_rabbitmq
from core.storage.neo4j_handler import initialize_neo4j, cleanup_neo4j
from core.storage.s3_handler import get_global_s3_handler, cleanup_global_s3_handler

logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL), format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Initializing Neo4j connection and indexes...")
    await initialize_neo4j()
    logging.info("Initializing global S3 handler...")
    await get_global_s3_handler()
    logging.info("Ensuring RabbitMQ connection...")
    await ensure_rmq_connection()
    orchestrator = get_global_orchestrator()
    register_processors(orchestrator)
    consumer_task = asyncio.create_task(rabbitmq_handler.start_consuming(orchestrator))
    logging.info("RabbitMQ consumer scheduler started successfully")
    try:
        yield
    finally:
        await cleanup_global_orchestrator()
        await rabbitmq_handler.stop_consuming()
        logging.info("RabbitMQ consumer stopped")
        await close_rabbitmq()
        logging.info("Cleaning up Neo4j connection...")
        await cleanup_neo4j()
        logging.info("Cleaning up global S3 handler...")
        await cleanup_global_s3_handler()
        consumer_task.cancel()
        try:
            await asyncio.wait_for(consumer_task, timeout=5.0)
        except asyncio.CancelledError:
            logging.info("Background tasks were cancelled during shutdown")
        except asyncio.TimeoutError:
            logging.warning("Background tasks didn't terminate within timeout")
        except Exception as exc:
            logging.error(f"Error waiting for background tasks: {exc}")

app = FastAPI(title=settings.API_TITLE, description=settings.API_DESCRIPTION, version=settings.API_VERSION, lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.include_router(api_router)

@app.get("/health")
async def health_check():
    return {"status": "ok", "timestamp": asyncio.get_event_loop().time()}
