import logging
from typing import Optional
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from config import settings

from core.db.base import Base

logger = logging.getLogger(__name__)

_global_db_session: Optional[AsyncSession] = None

SQLALCHEMY_DATABASE_URL = settings.DB_URL

if SQLALCHEMY_DATABASE_URL.startswith('sqlite'):
    engine = create_async_engine(
        SQLALCHEMY_DATABASE_URL.replace('sqlite:', 'sqlite+aiosqlite:'),
        echo=False,
        future=True
    )
else:
    engine = create_async_engine(
        SQLALCHEMY_DATABASE_URL,
        echo=False,
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

def get_global_db_session() -> AsyncSession:
    global _global_db_session
    if _global_db_session is None:
        _global_db_session = AsyncSessionLocal()
    return _global_db_session

async def cleanup_global_db_session():
    global _global_db_session
    if _global_db_session:
        await _global_db_session.close()
        _global_db_session = None

async def init_db():
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise