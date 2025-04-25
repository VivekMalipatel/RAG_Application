import logging
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from app.config import settings

logger = logging.getLogger(__name__)

# SQLite URL - Will be replaced with PostgreSQL in the future
SQLALCHEMY_DATABASE_URL = settings.DB_URL

# Create async engine for SQLite 
# Note: For SQLite, we need to use a special URL format for async
if SQLALCHEMY_DATABASE_URL.startswith('sqlite'):
    engine = create_async_engine(
        SQLALCHEMY_DATABASE_URL.replace('sqlite:', 'sqlite+aiosqlite:'),
        echo=True,
        future=True
    )
else:
    # For future PostgreSQL support
    engine = create_async_engine(
        SQLALCHEMY_DATABASE_URL,
        echo=True,
        future=True
    )

# Base class for all models
Base = declarative_base()

# Async session factory
AsyncSessionLocal = sessionmaker(
    engine, 
    expire_on_commit=False, 
    class_=AsyncSession
)

async def get_db():
    """Dependency for getting async database session"""
    db = AsyncSessionLocal()
    try:
        yield db
    finally:
        await db.close()

async def init_db():
    """Initialize the database"""
    try:
        # Create all tables
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise