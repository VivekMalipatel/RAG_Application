from app.api.db import Base, AsyncSessionLocal, engine
from sqlalchemy.future import select
from app.api.models import Document
from app.api.core.crud import CRUDBase
from app.api.core.security import create_access_token

class DocumentHandler:
    """Handles database interactions using SQLAlchemy ORM."""
    
    def __init__(self):
        self.document_crud = CRUDBase(Document)

    async def insert_file_metadata(self, user_id: str, file_name: str, file_path: str, file_size: int, file_hash: str, mime_type: str):
        """Inserts file metadata into the database using CRUD operations."""
        async with AsyncSessionLocal() as session:
            file_data = {
                "user_id": user_id,
                "file_name": file_name,
                "file_path": file_path,
                "file_size": file_size,
                "file_hash": file_hash,
                "mime_type": mime_type
            }
            # Directly await the async create method
            new_file = await self.document_crud.create(db=session, obj_in=file_data)
            return new_file

    async def get_document_metadata(self, user_id: str, file_path: str):
        """Fetches file metadata from PostgreSQL using CRUD operations."""
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(Document).filter_by(user_id=int(user_id), file_path=file_path)
            )
            file_meta = result.scalar_one_or_none()
            return file_meta

    def generate_file_token(self, file_data: dict) -> str:
        """Generates a JWT token for the provided file metadata."""
        token = create_access_token(file_data)
        return token
