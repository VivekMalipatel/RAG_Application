from pydantic_settings import BaseSettings
from typing import List
import os

class Settings(BaseSettings):
    # Project Information
    PROJECT_NAME: str = "Agent API"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"
    
    # OpenAI/LLM Configuration
    OPENAI_API_KEY: str
    OPENAI_API_BASE: str
    
    # External API Keys and URLs
    TAVILY_API_KEY: str = "your-tavily-key"
    TAVILY_API_URL: str = "https://api.tavily.com"
    
    # DuckDuckGo API (for web search)
    DUCKDUCKGO_API_URL: str = "https://api.duckduckgo.com"
    
    # Custom API configurations (for tools)
    CUSTOM_API_KEY: str = "your-custom-api-key"
    CUSTOM_API_URL: str = "https://api.example.com"
    
    # Model settings
    DEFAULT_MODEL: str = "qwen2.5vl:7b-q8_0"
    SEARCH_MODEL: str = "qwen2.5:7b-instruct-q8_0"
    
    # Security
    SECRET_KEY: str = "your-secret-key-change-this-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    ALLOWED_HOSTS: List[str] = ["*"]
    
    # Application settings
    LOG_LEVEL: str = "info"
    ENVIRONMENT: str = "development"
    
    # Model Router (legacy)
    MODEL_ROUTER_BASE_URL: str
    MODEL_ROUTER_API_KEY: str
    
    # Indexer service
    INDEXER_API_BASE_URL: str = "http://localhost:8001"
    
    # Tool-specific API configurations
    TOOL_API_TIMEOUT: int = 30
    TOOL_API_MAX_RETRIES: int = 3
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
