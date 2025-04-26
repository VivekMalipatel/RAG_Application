from sqlalchemy import Column, String, Text, ForeignKey
from sqlalchemy.orm import relationship

from app.models.queue_item import Base

class TextData(Base):
    __tablename__ = "text_data"
    
    queue_id = Column(String, ForeignKey("queue_items.id"), primary_key=True)
    content = Column(Text, nullable=False)
    
    queue_item = relationship("QueueItem", back_populates="text_data")