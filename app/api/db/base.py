from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import text
from app.config import settings
import logging

# Create Async Engine
engine = create_async_engine(settings.DATABASE_URL, future=True, echo=True)

# Create Async Session Factory
AsyncSessionLocal = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

# Base Model
Base = declarative_base()

# Dependency for DB session (to be used in FastAPI endpoints)
async def get_db():
    """Yields a new database session for each request."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

async def init_db():
    """Initializes the database on startup."""
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logging.info("Database initialized successfully")
    except Exception as e:
        logging.error(f"Database initialization failed: {e}")
        raise e

async def close_db():
    """Disposes of the database connection on FastAPI shutdown."""
    await engine.dispose()