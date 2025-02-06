from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Enum
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
import enum
from app.api.db.base import Base


class QuadrantStatus(str, enum.Enum):
    Q1 = "Q1"  # Important & Urgent
    Q2 = "Q2"  # Important & Not Urgent
    Q3 = "Q3"  # Not Important & Urgent
    Q4 = "Q4"  # Not Important & Not Urgent
    UNASSIGNED = "UNASSIGNED"

class Document(Base):
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    file_url = Column(String(500), nullable=False)
    file_type = Column(String(50), nullable=False)
    description = Column(Text, nullable=True)
    quadrant_status = Column(
        Enum(QuadrantStatus), 
        nullable=False, 
        default=QuadrantStatus.UNASSIGNED
    )
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    user = relationship("User", back_populates="documents")