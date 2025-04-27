from sqlalchemy import Column, String, LargeBinary, ForeignKey
from sqlalchemy.orm import relationship

from app.db.base import Base

class FileData(Base):
    __tablename__ = "file_data"
    
    queue_id = Column(String, ForeignKey("queue_items.id"), primary_key=True)
    content = Column(LargeBinary, nullable=False)
    
    queue_item = relationship("QueueItem", back_populates="file_data")