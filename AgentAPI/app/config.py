import os
from typing import Dict, Any, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    API Settings loaded from environment variables and .env file
    """
    # Application settings
    APP_NAME: str = "AgentAPI"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = True
    
    # Database settings
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///agent.db")
    
    # Model Router API settings
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "test-key")
    OPENAI_API_BASE: str = os.getenv("OPENAI_API_BASE", "http://localhost:8000/v1")

    # Memory settings
    MEMORY_EXPIRY_DAYS: int = 30
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )
    
settings = Settings()