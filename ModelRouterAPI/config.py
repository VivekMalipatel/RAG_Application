import os
from typing import Dict, List, Optional, Union
from pydantic_settings import BaseSettings
import secrets

class Settings(BaseSettings):
    API_V1_STR: str = "/v1"
    PROJECT_NAME: str = "ModelRouter API"
    
    SECRET_KEY: str = secrets.token_urlsafe(32)
    API_KEY_HEADER: str = "X-Api-Key"
    BEARER_TOKEN_HEADER: str = "Authorization"
    API_KEYS: List[str] = ["test-key"]
    USE_BEARER_TOKEN: bool = True
    
    DATABASE_URL: str = "sqlite:///./modelrouter.db"
    
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    OPENAI_BASE_URL: Optional[str] = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    OPENAI_ORG_ID: Optional[str] = os.getenv("OPENAI_ORG_ID")
    
    HUGGINGFACE_API_TOKEN: Optional[str] = os.getenv("HUGGINGFACE_API_TOKEN")
    
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_NUM_CTX: int = 8096

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()