from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Index
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from app.db.db import Base

class ThreadMemory(Base):
    __tablename__ = "thread_memory"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    thread_id = Column(String, index=True, nullable=False)
    agent_id = Column(String, ForeignKey("agent_memory.id"), nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    content_type = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    
    agent = relationship("AgentMemory", back_populates="thread_memories")
    
    __table_args__ = (
        Index("ix_thread_memory_thread_id_timestamp", "thread_id", "timestamp"),
    )

class AgentMemory(Base):
    __tablename__ = "agent_memory"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    agent_id = Column(String, index=True, nullable=False)
    client_id = Column(String, ForeignKey("client_memory.id"), nullable=True)
    name = Column(String, nullable=False)
    role = Column(String, nullable=False)
    instructions = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    capabilities = Column(Text, nullable=True)
    
    thread_memories = relationship("ThreadMemory", back_populates="agent")
    client = relationship("ClientMemory", back_populates="agents")

class ClientMemory(Base):
    __tablename__ = "client_memory"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    client_name = Column(String, nullable=False)
    api_key = Column(String, unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    settings = Column(Text, nullable=True)
    
    agents = relationship("AgentMemory", back_populates="client")

class UserMemory(Base):
    __tablename__ = "user_memory"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, index=True, nullable=False)
    client_id = Column(String, ForeignKey("client_memory.id"), nullable=False)
    username = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    preferences = Column(Text, nullable=True)
    
    client = relationship("ClientMemory")
    
    __table_args__ = (
        Index("ix_user_memory_client_id_user_id", "client_id", "user_id"),
    )