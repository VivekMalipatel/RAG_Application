from pydantic_settings import BaseSettings

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
    MINIO_ACCESS_KEY: str
    MINIO_SECRET_KEY: str
    MINIO_BUCKET_NAME: str
    QDRANT_URL: str
    OLLAMA_URL: str
    REDIS_CHAT_QUEUE: str
    REDIS_STANDARD_QUEUE: str
    MINIO_WEBHOOK_PATH: str 
    MINIO_WEBHOOK_SECRET: str
    PROJECT_NAME: str = "FastAPI User Management"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    class Config:
        env_file = ".env"
        extra = "allow"

settings = Settings()