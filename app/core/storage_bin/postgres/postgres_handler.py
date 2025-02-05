import logging
import uuid
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, String, Integer, DateTime, JSON, Text, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError

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

class MultipartUpload(Base):
    __tablename__ = "multipart_uploads"

    upload_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    uploadapproval_id = Column(String, primary_key=True)  # Composite primary key
    file_name = Column(String, nullable=False)
    user_id = Column(String, nullable=False)
    relative_path = Column(String, nullable=False)
    file_size = Column(Integer, nullable=False)
    mime_type = Column(String, nullable=False)
    total_chunks = Column(Integer, nullable=False)
    uploaded_chunks = Column(JSON, nullable=False, default={})  # Storing chunk details as JSON
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


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
    
    async def insert_multipart_upload(self, user_id, file_name, upload_id, uploadapproval_id, relative_path, file_size, mime_type, total_chunks, uploaded_chunks):
        """Stores initial multipart upload metadata in PostgreSQL."""
        async with self.async_session() as session:
            try:
                new_upload = MultipartUpload(
                    upload_id=upload_id,  # Store MinIO upload ID
                    uploadapproval_id=uploadapproval_id,
                    file_name=file_name,
                    user_id=user_id,
                    relative_path=relative_path,
                    file_size=file_size,
                    mime_type=mime_type,
                    total_chunks=total_chunks,
                    uploaded_chunks=uploaded_chunks,  
                )
                session.add(new_upload)
                await session.commit() # Commit transaction asynchronously
                await session.refresh(new_upload) # Refresh object with DB-generated values
                return True
            except SQLAlchemyError as e:
                logging.error(f"Error initializing multipart upload for '{file_name}': {e}")
                await session.rollback()
                return False
            
    async def delete_multipart_upload(self, upload_id: str):
        """Deletes multipart upload details from PostgreSQL."""
        async with self.async_session() as session:
            try:
                # Query to find the record
                result = await session.execute(select(MultipartUpload).filter_by(upload_id=upload_id))
                multipart_upload = result.scalars().first()
    
                if multipart_upload:
                    await session.delete(multipart_upload)  # Delete the found record
                    await session.commit()  # Commit the deletion
                else:
                    logging.warning(f"No multipart upload found with upload_id: {upload_id}")
            except SQLAlchemyError as e:
                logging.error(f"Error deleting multipart upload with upload_id '{upload_id}': {e}")
                await session.rollback()  # Rollback transaction if error occurs
            
    async def get_multipart_upload(self, uploadapproval_id: str):
        """Fetches multipart upload details from PostgreSQL."""

        async with self.async_session() as session:
            try:
                stmt = select(MultipartUpload).filter_by(uploadapproval_id=uploadapproval_id)
                result = await session.execute(stmt)
                uploads = result.scalars().first()  # Get all matching records
                
                if not uploads:
                    logging.warning(f"No multipart uploads found for uploadapproval_id: {uploadapproval_id}")
                    
                return uploads  # Returns a list of matching records
            except SQLAlchemyError as e:
                logging.error(f"Error fetching multipart uploads for uploadapproval_id '{uploadapproval_id}': {e}")
                return None
            
    async def update_multipart_upload(self, uploadapproval_id: str, uploaded_chunks: dict, etag):
        