import uuid
from datetime import datetime
from sqlalchemy import Column, String, DateTime, JSON, Text, Enum
from app.db.database import Base

class FileItem(Base):
    __tablename__ = "file_items"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String, nullable=True)
    file_type = Column(String, nullable=True)
    url = Column(String, nullable=True)
    raw_text = Column(Text, nullable=True)
    source = Column(String, nullable=False)
    metadata = Column(JSON, nullable=True)
    status = Column(
        Enum("queued", "processing", "completed", "failed", name="status_enum"),
        default="queued"
    )
    status_message = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    storage_path = Column(String, nullable=True)
    embedding_stored = Column(String, nullable=True)