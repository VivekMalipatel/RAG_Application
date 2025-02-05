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
            
    async def update_multipart_part(self, upload_id: str, chunk_number: int , etag: str):
        """Apprehends the part_info into uploaded chunks for a multipart upload."""
        async with self.async_session() as session:
            try:
                result = await session.execute(
                    select(MultipartUpload).where(MultipartUpload.upload_id == upload_id)
                )
                upload_session = result.scalar_one_or_none()

                if not upload_session:
                    logging.error(f"Multipart upload '{upload_id}' not found.")
                    return False
                
                upload_session.uploaded_chunks[str(chunk_number)] = etag

                await session.commit()
                logging.info(f"Part {chunk_number} updated for upload ID '{upload_id}' with ETag: {etag}")
                return True

            except SQLAlchemyError as e:
                logging.error(f"Error updating part {chunk_number} for upload ID '{upload_id}': {e}")
                await session.rollback()
                return False