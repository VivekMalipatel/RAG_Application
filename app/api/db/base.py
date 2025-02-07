from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from app.api.core.config import Settings

# Create Async Engine
engine = create_async_engine(Settings.DATABASE_URL, future=True, echo=True)

# Create Async Session Factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, class_=AsyncSession)

# Base Model
Base = declarative_base()

# Dependency for DB session (to be used in FastAPI endpoints)
async def get_db():
    async with SessionLocal() as session:
        yield session