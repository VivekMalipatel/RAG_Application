import logging
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from app.config import settings

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

Base = declarative_base()

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