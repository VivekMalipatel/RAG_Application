import logging
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from app.config import settings

from app.db.base import Base
from app.models.queue_item import QueueItem
from app.models.file_data import FileData
from app.models.text_data import TextData
from app.models.url_data import URLData
from app.models.failure_queue_item import FailureQueueItem

logger = logging.getLogger(__name__)

SQLALCHEMY_DATABASE_URL = settings.DB_URL

if SQLALCHEMY_DATABASE_URL.startswith('sqlite'):
    engine = create_async_engine(
        SQLALCHEMY_DATABASE_URL.replace('sqlite:', 'sqlite+aiosqlite:'),
        echo=True,
        future=True
    )
else:
    engine = create_async_engine(
        SQLALCHEMY_DATABASE_URL,
        echo=True,
        future=True
    )

AsyncSessionLocal = sessionmaker(
    engine, 
    expire_on_commit=False, 
    class_=AsyncSession
)

async def get_db():
    db = AsyncSessionLocal()
    try:
        yield db
    finally:
        await db.close()

async def init_db():
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise