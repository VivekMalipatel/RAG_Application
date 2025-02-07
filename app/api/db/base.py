from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import text
from app.config import settings
import logging

# Create Async Engine
engine = create_async_engine(settings.DATABASE_URL, future=True, echo=True)

# Create Async Session Factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, class_=AsyncSession)

# Base Model
Base = declarative_base()

# Dependency for DB session (to be used in FastAPI endpoints)
async def get_db():
    async with SessionLocal() as session:
        yield session

async def init_db():
    """
    Initializes the database by testing the connection.
    """
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logging.info("Database connection test succeeded.")
    except Exception as e:
        logging.error(f"Database initialization failed: {e}")
        raise e

async def close_db():
    """Closes the database connection on FastAPI shutdown."""
    await engine.dispose()