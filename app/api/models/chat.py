from sqlalchemy import Column, String, DateTime, JSON, Boolean
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import uuid
from app.api.db.base import Base

class Chat(Base):
    __tablename__ = "chat_messages"

    index = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chat_id = Column(UUID(as_uuid=True), default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), index=True, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    message = Column(String, nullable=False)
    message_type = Column(String, nullable=False)  # "user" or "agent"
    chat_metadata = Column(JSON, nullable=True)         # Additional data
    entities = Column(JSON, nullable=True)         # Extracted entities
    relationships = Column(JSON, nullable=True)        # Relationships (for Neo4j)
    chat_summary = Column(String, nullable=True)   # Summary of the chat