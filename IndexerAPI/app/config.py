import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # API configurations
    API_TITLE: str = "File Indexer API"
    API_DESCRIPTION: str = "API for indexing various file types for RAG applications"
    API_VERSION: str = "0.1.0"
    
    # Database
    DB_URL: str = os.getenv("DB_URL", "sqlite:///./app.db")
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

settings = Settings()