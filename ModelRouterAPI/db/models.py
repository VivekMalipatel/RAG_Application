from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Boolean, Text
from sqlalchemy.orm import relationship
import datetime

from db.base import Base

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