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
    
    usage = relationship("Usage", back_populates="api_key")


class Usage(Base):
    __tablename__ = "usage"

    id = Column(Integer, primary_key=True, index=True)
    api_key_id = Column(Integer, ForeignKey("api_keys.id"))
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    endpoint = Column(String)
    model = Column(String)
    provider = Column(String)
    prompt_tokens = Column(Integer)
    completion_tokens = Column(Integer, default=0)
    total_tokens = Column(Integer)
    processing_time = Column(Float)
    request_id = Column(String, unique=True, index=True)
    request_data = Column(Text, nullable=True)
    
    api_key = relationship("ApiKey", back_populates="usage")
    
    __table_args__ = (
        Index("idx_api_key_timestamp", "api_key_id", "timestamp"),
    )


class AvailableModel(Base):
    __tablename__ = "available_models"
    
    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(String, unique=True, index=True)
    provider = Column(Enum(Provider))
    model_type = Column(String)
    created = Column(Integer)
    last_seen = Column(Integer)
    is_available = Column(Boolean, default=True)
    
    __table_args__ = (
        Index("idx_model_type_available", "model_type", "is_available"),
    )