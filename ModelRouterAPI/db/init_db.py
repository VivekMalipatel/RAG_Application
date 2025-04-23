import logging
from sqlalchemy.orm import Session
import uuid

from db.base import Base, engine
from db.session import SessionLocal
from db.models import ApiKey, Usage
from config import settings

# Get logger
logger = logging.getLogger(__name__)

def init_db() -> None:
    """Initialize the database."""
    # Create tables
    Base.metadata.create_all(bind=engine)
    
    # Create a database session
    db = SessionLocal()
    try:
        # Check if there are any API keys
        existing_keys = db.query(ApiKey).count()
        
        # Only create a default API key if none exist
        if existing_keys == 0:
            for api_key in settings.API_KEYS:
                new_key = ApiKey(
                    key=api_key,
                    user_id="admin",
                    name="Default API Key",
                    is_active=True
                )
                db.add(new_key)
            
            db.commit()
            logger.info("Created default API key(s)")
        else:
            logger.info(f"Database already has {existing_keys} API keys")
            
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        db.rollback()
        raise
    finally:
        db.close()
        
    logger.info("Database initialization complete")