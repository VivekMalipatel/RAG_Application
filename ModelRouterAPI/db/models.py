from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Boolean, Text, Enum, Index
from sqlalchemy.orm import relationship
import datetime

from db.base import Base
from model_provider import Provider

class ApiKey(Base):
    __tablename__ = "api_keys"

    id = Column(Integer, primary_key=True, index=True)
    key = Column(String, unique=True, index=True)
    user_id = Column(String, index=True)
    name = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    # Relationship to usage records
    usage = relationship("Usage", back_populates="api_key")


class Usage(Base):
    __tablename__ = "usage"

    id = Column(Integer, primary_key=True, index=True)
    api_key_id = Column(Integer, ForeignKey("api_keys.id"))
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    endpoint = Column(String)  # chat, completions, embeddings
    model = Column(String)
    provider = Column(String)  # openai, huggingface, ollama
    prompt_tokens = Column(Integer)
    completion_tokens = Column(Integer, default=0)
    total_tokens = Column(Integer)
    processing_time = Column(Float)  # in seconds
    request_id = Column(String, unique=True, index=True)
    request_data = Column(Text, nullable=True)  # JSON string of the request
    
    # Relationship to API key
    api_key = relationship("ApiKey", back_populates="usage")
    
    # Add index for common queries
    __table_args__ = (
        Index("idx_api_key_timestamp", "api_key_id", "timestamp"),
    )


class AvailableModel(Base):
    """Track available models from different providers"""
    __tablename__ = "available_models"
    
    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(String, unique=True, index=True)  # The model ID/name
    provider = Column(Enum(Provider))  # Which provider this model belongs to
    model_type = Column(String)  # TEXT_GENERATION, TEXT_EMBEDDING, RERANKER
    created = Column(Integer)  # When this model was first created/discovered
    last_seen = Column(Integer)  # Last time this model was confirmed available
    is_available = Column(Boolean, default=True)  # Whether this model is currently available
    
    # Add index for fast lookup by type and availability
    __table_args__ = (
        Index("idx_model_type_available", "model_type", "is_available"),
    )