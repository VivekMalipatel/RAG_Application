from sqlalchemy import Column, String, ForeignKey, Text, DateTime
from sqlalchemy.orm import relationship

from app.db.base import Base

class FailureQueueItem(Base):
    __tablename__ = "failure_queue_items"
    
    queue_id = Column(String, ForeignKey("queue_items.id"), primary_key=True)
    error_message = Column(Text, nullable=False)
    failure_datetime = Column(DateTime, nullable=False)
    retry_count = Column(String, default=0)
    
    queue_item = relationship("QueueItem", backref="failure_queue_item")