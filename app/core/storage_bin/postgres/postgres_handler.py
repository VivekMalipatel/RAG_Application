import logging
import uuid
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, String, Integer, DateTime, JSON, Text, select

# SQLAlchemy Base Model
Base = declarative_base()

class FileMetadata(Base):
    """ORM Model for file metadata storage."""
    __tablename__ = "files"

    file_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, nullable=False)
    file_name = Column(Text, nullable=False)
    file_path = Column(Text, nullable=False)
    file_size = Column(Integer, nullable=False)
    file_hash = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

# class MultipartUpload(Base):
#     __tablename__ = "multipart_uploads"

#     upload_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
#     file_id = Column(String, nullable=False)
#     user_id = Column(String, nullable=False)
#     status = Column(String, nullable=False)
#     parts = Column(JSON, nullable=False)
#     created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
#     updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


class PostgresHandler:
    """Handles database interactions using SQLAlchemy ORM."""

    def __init__(self, db_url: str):
        """
        Initializes PostgreSQL connection using SQLAlchemy async engine.

        Args:
            db_url (str): PostgreSQL connection URL.
        """
        self.engine = create_async_engine(db_url, echo=False, future=True)
        self.async_session = sessionmaker(
            bind=self.engine, expire_on_commit=False, class_=AsyncSession
        )
    
    async def init_db(self):
        """Initializes the database (creates tables if they do not exist)."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logging.info("PostgreSQL tables initialized.")
    
    async def close(self):
        """Closes the database engine."""
        await self.engine.dispose()
        logging.info("PostgreSQL connection closed.")
 
 
    # -------------------------------
    # FILE METADATA CRUD OPERATIONS
    # -------------------------------

    async def insert_file_metadata(self, user_id: str, file_name: str, file_path: str, file_size: int, file_hash: str):
        """Inserts file metadata into the database."""
        async with self.async_session() as session:
            try:
                new_file = FileMetadata(
                    user_id=user_id,
                    file_name=file_name,
                    file_path=file_path,
                    file_size=file_size,
                    file_hash=file_hash
                )
                session.add(new_file)
                await session.commit()
                await session.refresh(new_file)
                return new_file
            except Exception as e:
                logging.error(f"Error inserting file metadata for '{file_name}': {e}")
                await session.rollback()
                return None

    async def get_file_metadata(self, user_id: str, file_name: str):
        """Fetches file metadata from PostgreSQL."""
        async with self.async_session() as session:
            try:
                stmt = select(FileMetadata).filter_by(user_id=user_id, file_name=file_name)
                result = await session.execute(stmt)
                return result.scalars().first()
            except Exception as e:
                logging.error(f"Error fetching file metadata for '{file_name}': {e}")
                return None