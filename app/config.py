from pydantic_settings import BaseSettings
from typing import Set

class Settings(BaseSettings):
    APP_HOST: str
    APP_PORT: int
    DEBUG_MODE: bool
    LOG_LEVEL: str
    DATABASE_URL: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    REDIS_URL: str
    SECRET_KEY: str
    ALGORITHM: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int
    MINIO_ENDPOINT: str
    MINIO_ROOT_USER: str
    MINIO_ROOT_PASSWORD: str
    MINIO_ACCESS_KEY: str
    MINIO_SECRET_KEY: str
    MINIO_BUCKET_NAME: str
    QDRANT_URL: str
    OLLAMA_URL: str
    REDIS_CHAT_QUEUE: str
    REDIS_STANDARD_QUEUE: str
    MINIO_WEBHOOK_PATH: str 
    MINIO_WEBHOOK_SECRET: str
    GGUF_CONVERTER_URL: str
    OPENAI_API_KEY: str
    OPENAI_API_URL: str
    HUGGINGFACE_TOKEN: str
    TEXT_EMBEDDING_SOURCE: str
    TEXT_EMBEDDING_MODEL: str
    TEXT_CHUNK_SIZE: str
    TEXT_DOC_CONTEXT_SIZE: str
    TEXT_CHUNK_CONTEXT_SIZE: str
    TEXT_LLM_MODEL: str
    TEXT_LLM_QUANTIZATION: str
    TEXT_LLM_TEMPERATURE: str
    TEXT_LLM_TOP_P: str
    TEXT_LLM_MAX_TOKENS: str
    TEXT_CHUNK_OVERLAP: str
    PROJECT_NAME: str = "FastAPI User Management"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    TEXT_MIME_TYPES: str = ""
    IMAGE_MIME_TYPES: str = ""
    AUDIO_MIME_TYPES: str = ""
    VIDEO_MIME_TYPES: str = ""
    MULTIMODAL_MIME_TYPES: str = ""
    STRUCTURED_MIME_TYPES: str = ""

    class Config:
        env_file = ".env"
        extra = "allow"
    
    def load_mime_types(self):
        """Load MIME type lists from comma-separated env variables."""
        self.TEXT_MIME_TYPES = set(self.TEXT_MIME_TYPES.split(","))
        self.IMAGE_MIME_TYPES = set(self.IMAGE_MIME_TYPES.split(","))
        self.AUDIO_MIME_TYPES = set(self.AUDIO_MIME_TYPES.split(","))
        self.VIDEO_MIME_TYPES = set(self.VIDEO_MIME_TYPES.split(","))
        self.MULTIMODAL_MIME_TYPES = set(self.MULTIMODAL_MIME_TYPES.split(","))
        self.STRUCTURED_MIME_TYPES = set(self.STRUCTURED_MIME_TYPES.split(","))
        
settings = Settings()
settings.load_mime_types()