from sqlalchemy import Column, String, DateTime, Text
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class QueueItem(Base):
    __tablename__ = "queue_items"
    
    id = Column(String, primary_key=True)
    source = Column(String, nullable=False)
    item_type = Column(String, nullable=False)
    status = Column(String, nullable=False)
    indexing_datetime = Column(DateTime, nullable=False)
    metadata = Column(Text, nullable=True)
    message = Column(Text, nullable=True)
    
    file_data = relationship("FileData", back_populates="queue_item", uselist=False, cascade="all, delete-orphan")
    text_data = relationship("TextData", back_populates="queue_item", uselist=False, cascade="all, delete-orphan")
    url_data = relationship("URLData", back_populates="queue_item", uselist=False, cascade="all, delete-orphan")