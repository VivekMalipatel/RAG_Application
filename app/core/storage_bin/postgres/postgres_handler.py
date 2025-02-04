import logging
import uuid
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, String, Integer, DateTime, JSON, Text, select

# SQLAlchemy Base Model
Base = declarative_base()


# -------------------------------
# PostgreSQL ORM Models
# -------------------------------
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
    """ORM Model for tracking multipart uploads."""
    __tablename__ = "multipart_uploads"

    upload_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, nullable=False)
    file_name = Column(Text, nullable=False)
    total_parts = Column(Integer, nullable=False)
    uploaded_parts = Column(JSON, default={})  # Stores part_number -> ETag mapping
    status = Column(String, default="in-progress")  # 'in-progress', 'completed', 'failed'
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


# -------------------------------
# PostgreSQL Handler Class
# -------------------------------
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
        """Closes the database connection."""
        await self.engine.dispose()
        logging.info("PostgreSQL connection closed.")

    # -------------------------------
    # FILE METADATA CRUD OPERATIONS
    # -------------------------------

    async def insert_file_metadata(self, user_id: str, file_name: str, file_path: str, file_size: int, file_hash: str):
        """Stores file metadata in PostgreSQL."""
        async with self.async_session() as session:
            try:
                new_file = FileMetadata(
                    user_id=user_id,
                    file_name=file_name,
                    file_path=file_path,
                    file_size=file_size,
                    file_hash=file_hash,
                )
                session.add(new_file)
                await session.commit()
                logging.info(f"File '{file_name}' metadata stored successfully. File ID: {new_file.file_id}")
                return new_file.file_id
            except Exception as e:
                logging.error(f"Error inserting file metadata for '{file_name}': {e}")
                await session.rollback()
                return None

    async def get_file_metadata(self, user_id: str, file_name: str):
        """Fetches file metadata from PostgreSQL."""
        async with self.async_session() as session:
            try:
                result = await session.execute(
                    select(FileMetadata).where(
                        (FileMetadata.user_id == user_id) & (FileMetadata.file_name == file_name)
                    )
                )
                record = result.scalar_one_or_none()
                return record if record else None
            except Exception as e:
                logging.error(f"Error fetching metadata for '{file_name}': {e}")
                return None

    # -------------------------------
    # MULTIPART UPLOAD TRACKING
    # -------------------------------

    async def insert_multipart_upload(self, user_id: str, file_name: str, total_parts: int, upload_id: str):
        """Stores initial multipart upload metadata in PostgreSQL."""
        async with self.async_session() as session:
            try:
                new_upload = MultipartUpload(
                    upload_id=upload_id,  # Store MinIO upload ID
                    user_id=user_id,
                    file_name=file_name,
                    total_parts=total_parts,
                    uploaded_parts={},  # No parts uploaded yet
                    status="in-progress"
                )
                session.add(new_upload)
                await session.commit()
                logging.info(f"Multipart upload started for '{file_name}', Upload ID: {upload_id}")
                return upload_id
            except Exception as e:
                logging.error(f"Error initializing multipart upload for '{file_name}': {e}")
                await session.rollback()
                return None

    async def update_multipart_part(self, upload_id: str, part_number: int, etag: str):
        """Updates PostgreSQL with the uploaded part number and its ETag."""
        async with self.async_session() as session:
            try:
                result = await session.execute(
                    select(MultipartUpload).where(MultipartUpload.upload_id == upload_id)
                )
                upload_session = result.scalar_one_or_none()

                if not upload_session:
                    logging.error(f"Multipart upload '{upload_id}' not found.")
                    return False

                uploaded_parts = upload_session.uploaded_parts or {}
                uploaded_parts[part_number] = etag
                upload_session.uploaded_parts = uploaded_parts

                await session.commit()
                logging.info(f"Part {part_number} updated for upload ID '{upload_id}' with ETag: {etag}")
                return True
            except Exception as e:
                logging.error(f"Error updating part {part_number} for upload ID '{upload_id}': {e}")
                await session.rollback()
                return False

    async def complete_multipart_upload(self, upload_id: str):
        """Marks a multipart upload as complete in PostgreSQL."""
        async with self.async_session() as session:
            try:
                result = await session.execute(
                    select(MultipartUpload).where(MultipartUpload.upload_id == upload_id)
                )
                upload_session = result.scalar_one_or_none()

                if not upload_session:
                    logging.error(f"Multipart upload '{upload_id}' not found.")
                    return False

                if len(upload_session.uploaded_parts) != upload_session.total_parts:
                    logging.error(f"Upload '{upload_id}' incomplete. Missing parts.")
                    return False

                upload_session.status = "completed"
                await session.commit()
                logging.info(f"Upload '{upload_id}' marked as completed in PostgreSQL.")
                return True
            except Exception as e:
                logging.error(f"Error completing multipart upload '{upload_id}': {e}")
                await session.rollback()
                return False

    async def cancel_multipart_upload(self, upload_id: str):
        """Cancels a multipart upload and removes its record from PostgreSQL."""
        async with self.async_session() as session:
            try:
                result = await session.execute(
                    select(MultipartUpload).where(MultipartUpload.upload_id == upload_id)
                )
                upload_session = result.scalar_one_or_none()

                if not upload_session:
                    logging.error(f"Multipart upload '{upload_id}' not found.")
                    return False

                await session.delete(upload_session)
                await session.commit()
                logging.info(f"Multipart upload '{upload_id}' canceled and removed from PostgreSQL.")
                return True
            except Exception as e:
                logging.error(f"Error canceling multipart upload '{upload_id}': {e}")
                await session.rollback()
                return False