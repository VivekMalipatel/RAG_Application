from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Database:
    def __init__(self, database_url: str):
        self.engine = create_engine(database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.Base = declarative_base()

    def get_db(self):
        """Dependency function to get a database session."""
        db = self.SessionLocal()
        try:
            yield db
        finally:
            db.close()

    def init_db(self):
        """Creates tables in the database."""
        self.Base.metadata.create_all(bind=self.engine)

if __name__ == "__main__":
    database_url = os.getenv("DATABASE_URL")
    db_instance = Database(database_url)
    db_instance.init_db()
    print("✅ Database tables created successfully!")