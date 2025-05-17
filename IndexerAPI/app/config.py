import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    API_TITLE: str = "File Indexer API"
    API_DESCRIPTION: str = "API for indexing various file types for RAG applications"
    API_VERSION: str = "0.1.0"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    DB_URL: str = os.getenv("DB_URL", "sqlite:///./app.db")

    UNOSERVER_HOST: str = os.getenv("UNOSERVER_HOST", "unoserver")
    UNOSERVER_PORT: int = int(os.getenv("UNOSERVER_PORT", 2003))


settings = Settings()