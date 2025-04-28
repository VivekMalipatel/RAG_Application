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

    DEFAULT_MODELS: Dict[str, str] = {
        "openai": "gpt-3.5-turbo",
        "ollama": "llama2",
        "huggingface": "mistralai/Mistral-7B-Instruct-v0.2"
    }
    
    AUTO_SELECT_MODEL: bool = True
    PROVIDER_PRIORITY: List[str] = ["openai", "ollama", "huggingface"]
    COST_SENSITIVE_MODE: bool = False
    
    ENABLE_VISION: bool = True
    ENABLE_AUDIO: bool = False
    
    ENABLE_FUNCTION_CALLING: bool = True
    FUNCTION_SCHEMA_PATH: Optional[str] = None

    ENABLE_MODEL_CACHE: bool = True
    CACHE_EXPIRY_SECONDS: int = 3600
    
    RATE_LIMIT_ENABLED: bool = True
    REQUESTS_PER_MINUTE: int = 60
    
    PREFERRED_DEVICE: str = os.getenv("PREFERRED_DEVICE", "auto")
    CUDA_VISIBLE_DEVICES: str = os.getenv("CUDA_VISIBLE_DEVICES", "")
    MPS_FALLBACK_TO_CPU: bool = True
    DEVICE_MAP_STRATEGY: str = "auto"
    LOW_CPU_MEM_USAGE: bool = False

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()