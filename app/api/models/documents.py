from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Enum
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
import enum
from app.api.db.base import Base


class QuadrantStatus(str, enum.Enum):
    Q1 = "Not Processed"  # Not Processed
    Q2 = "Processing"  # Important & Not Urgent
    Q3 = "Priority Processed"  # Not Important & Urgent
    Q4 = "Fully Processed"  # Not Important & Not Urgent
    Q5 = "Failed Priority Processing"  # Important & Urgent
    Q6 = "Failed Full Processing"  # Important & Urgent


class Document(Base):
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    file_name = Column(Text, nullable=False)
    file_path = Column(Text, nullable=False) #TODO: make it unique
    mime_type = Column(String, nullable=False)
    file_size = Column(Integer, nullable=False)
    file_hash = Column(Text, nullable=False)
    description = Column(Text, nullable=True)
    quadrant_status = Column(
        Enum(QuadrantStatus), 
        nullable=False, 
        default=QuadrantStatus.Q1
    )
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    user = relationship("User", back_populates="documents")