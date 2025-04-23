from typing import Generator

from sqlalchemy.orm import Session

from db.base import SessionLocal

def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that provides a SQLAlchemy session.
    Ensures the session is closed after the request is processed.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()