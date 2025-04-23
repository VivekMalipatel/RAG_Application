import os
from typing import Dict, List, Optional, Union
from pydantic_settings import BaseSettings
import secrets

class Settings(BaseSettings):
    # API settings
    API_V1_STR: str = "/v1"
    PROJECT_NAME: str = "ModelRouter API"
    
    # Security
    SECRET_KEY: str = secrets.token_urlsafe(32)
    API_KEY_HEADER: str = "X-Api-Key"
    API_KEYS: List[str] = ["test-key"]  # Default test key, replace in production
    
    # Database settings
    DATABASE_URL: str = "sqlite:///./modelrouter.db"
    
    # OpenAI provider settings
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    OPENAI_BASE_URL: Optional[str] = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    
    # HuggingFace provider settings
    HUGGINGFACE_API_TOKEN: Optional[str] = os.getenv("HUGGINGFACE_API_TOKEN")
    
    # Ollama provider settings
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # Default models for each provider
    DEFAULT_MODELS: Dict[str, str] = {
        "openai": "gpt-3.5-turbo",
        "ollama": "llama2",
        "huggingface": "mistralai/Mistral-7B-Instruct-v0.2"
    }

    # Cache settings
    ENABLE_MODEL_CACHE: bool = True
    CACHE_EXPIRY_SECONDS: int = 3600  # 1 hour

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()