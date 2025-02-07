# Added from postgres_handler.py for handling PostgreSQL operations
import logging
import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Integer, DateTime, Text, select
from sqlalchemy.exc import SQLAlchemyError
from app.api.db import Base, SessionLocal, engine
from app.api.models import Document
from app.api.core.crud import CRUDBase
from app.api.core.security import create_access_token
from app.api.db.base import Base  # using base.py

class PostgresHandler:
    """Handles database interactions using SQLAlchemy ORM."""
    
    def __init__(self):
        pass

    async def init_db(self):
        """Initializes the database (creates tables if they do not exist)."""
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logging.info("PostgreSQL tables initialized.")

    async def close(self):
        """Closes the database engine."""
        await engine.dispose()
        logging.info("PostgreSQL connection closed.")

    async def insert_file_metadata(self, user_id: str, file_name: str, file_path: str, file_size: int, file_hash: str, mime_type: str):
        """Inserts file metadata into the database."""
        async with SessionLocal() as session:
            document_crud = CRUDBase(Document)
            file_data = {
                "user_id": user_id,
                "file_name": file_name,
                "file_path": file_path,
                "file_size": file_size,
                "file_hash": file_hash,
                "mime_type": mime_type
            }
            new_file = await session.run_sync(document_crud.create, obj_in=file_data)
            return new_file

    async def get_file_metadata(self, user_id: str, file_name: str):
        """Fetches file metadata from PostgreSQL."""
        async with SessionLocal() as session:
            file_meta = await session.run_sync(
                lambda s: s.query(Document).filter_by(user_id=user_id, file_name=file_name).first()
            )
            return file_meta

    def generate_file_token(self, file_data: dict) -> str:
        """Generates a JWT token for the provided file metadata."""
        token = create_access_token(file_data)
        return token