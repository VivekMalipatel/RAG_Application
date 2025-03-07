from pydantic_settings import BaseSettings
from app.core.models.model_provider import Provider

class Settings(BaseSettings):
    APP_HOST: str
    APP_PORT: int
    DEBUG_MODE: bool
    LOG_LEVEL: str

    #Postgres
    DATABASE_URL: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str

    #Redis
    REDIS_URL: str
    SECRET_KEY: str
    ALGORITHM: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int

    #Minio
    MINIO_ENDPOINT: str
    MINIO_ROOT_USER: str
    MINIO_ROOT_PASSWORD: str
    MINIO_ACCESS_KEY: str
    MINIO_SECRET_KEY: str
    MINIO_BUCKET_NAME: str
    #Minio-Webhook
    MINIO_WEBHOOK_PATH: str 
    MINIO_WEBHOOK_SECRET: str

    #Qdrant
    QDRANT_URL: str
    OLLAMA_URL: str

    #Redis Queues
    REDIS_CHAT_QUEUE: str
    REDIS_STANDARD_QUEUE: str

    #Neo4j
    NEO4J_URI: str
    NEO4J_USER: str
    NEO4J_PASSWORD: str
    NEO4J_DATABASE: str

    #GGUF_CONVERTER
    GGUF_CONVERTER_URL: str

    #Model Providers
    OPENAI_API_KEY: str
    OPENAI_API_URL: str
    HUGGINGFACE_TOKEN: str

    TEXT_MIME_TYPES: str = ""
    IMAGE_MIME_TYPES: str = ""
    AUDIO_MIME_TYPES: str = ""
    VIDEO_MIME_TYPES: str = ""
    MULTIMODAL_MIME_TYPES: str = ""
    STRUCTURED_MIME_TYPES: str = ""

    # TEXT PROCESSING CONFIGURATION
    TEXT_EMBEDDING_PROVIDER: Provider
    TEXT_EMBEDDING_MODEL_NAME: str
    
    TEXT_CHUNK_SIZE: int
    TEXT_CHUNK_OVERLAP: int
    
    TEXT_DOCUMENT_CONTEXT_MAX_TOKENS: int
    TEXT_CHUNK_CONTEXT_MAX_TOKENS: int
    
    TEXT_CONTEXT_LLM_PROVIDER: Provider
    TEXT_CONTEXT_LLM_MODEL_NAME: str
    TEXT_CONTEXT_LLM_QUANTIZATION: str
    TEXT_CONTEXT_LLM_TEMPERATURE: float
    TEXT_CONTEXT_LLM_TOP_P: float

    # CHAT LLM CONFIGURATION
    TEXT_LLM_MODEL_NAME: str
    TEXT_LLM_TEMPERATURE: float
    TEXT_LLM_TOP_P: float
    TEXT_LLM_QUANTIZATION: str
    TEXT_LLM_PROVIDER: Provider

    PROJECT_NAME: str = "FastAPI User Management"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

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