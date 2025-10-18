import os
from pathlib import Path
from dotenv import load_dotenv

env_candidates = [
    Path(__file__).resolve().parents[2] / ".env",
    Path(__file__).resolve().parents[3] / ".env",
    Path.cwd() / ".env",
]
for candidate in env_candidates:
    if candidate.exists():
        load_dotenv(dotenv_path=candidate)
        break
else:
    load_dotenv()

class Settings:
    API_TITLE: str = os.getenv("API_TITLE", "File Indexer API")
    API_DESCRIPTION: str = os.getenv("API_DESCRIPTION", "API for indexing various file types for RAG applications")
    API_VERSION: str = os.getenv("API_VERSION", "0.1.0")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    DB_URL: str | None = os.getenv("DB_URL")
    UNOSERVER_HOST: str = "0.0.0.0"
    UNOSERVER_PORT: int | None = int(os.getenv("UNOSERVER_PORT", "2003")) or None
    
    # OpenAI Embedding Configuration (for multimodal embeddings)
    EMBEDDING_API_KEY: str | None = os.getenv("EMBEDDING_API_KEY")
    EMBEDDING_API_BASE: str | None = os.getenv("EMBEDDING_API_BASE")
    EMBEDDING_MODEL: str | None = os.getenv("EMBEDDING_MODEL")
    
    # Azure OpenAI LLM Configuration
    AZURE_OPENAI_ENDPOINT: str | None = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_KEY: str | None = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_API_VERSION: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
    AZURE_OPENAI_DEPLOYMENT_LLM: str = os.getenv("AZURE_OPENAI_DEPLOYMENT_LLM", "gpt-5-mini")
    
    # OpenAI LLM Configuration (commented out - uncomment for full OpenAI)
    # LLM_API_KEY: str | None = os.getenv("LLM_API_KEY")
    # LLM_API_BASE: str | None = os.getenv("LLM_API_BASE")
    # LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o")
    
    EMBEDDING_DIMENSIONS: int = int(os.getenv("EMBEDDING_DIMENSIONS", "2048"))
    EMBEDDING_CLIENT_TIMEOUT: float = float(os.getenv("EMBEDDING_CLIENT_TIMEOUT", "3600"))
    LLM_CLIENT_TIMEOUT: float = float(os.getenv("LLM_CLIENT_TIMEOUT", "3600"))
    RETRIES: int = int(os.getenv("RETRIES", "3"))
    RETRY_DELAY: int = int(os.getenv("RETRY_DELAY", "2"))
    MINIO_ENDPOINT_URL: str | None = os.getenv("MINIO_ENDPOINT_URL")
    S3_BUCKET_NAME: str | None = os.getenv("S3_BUCKET_NAME")
    AWS_ACCESS_KEY_ID: str | None = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY: str | None = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_REGION: str | None = os.getenv("AWS_REGION")
    RABBITMQ_URL: str | None = os.getenv("RABBITMQ_URL")
    RABBITMQ_QUEUE_NAME: str | None = os.getenv("RABBITMQ_QUEUE_NAME")
    RABBITMQ_HEARTBEAT: int = int(os.getenv("RABBIT_MQ_HEARTBEAT", "600"))
    RABBITMQ_CONSUMER_TIMEOUT: int = int(os.getenv("RABBIT_MQ_CONSUMER_TIMEOUT", "300"))
    RABBITMQ_X_CONSUMER_TIMEOUT: int = int(os.getenv("RABBIT_MQ_X_CONSUMER_TIMEOUT", "86400000"))
    RABBITMQ_MESSAGE_GET_TIMEOUT: int = int(os.getenv("RABBIT_MQ_MESSAGE_GET_TIMEOUT", "30"))
    RABBITMQ_PREFETCH_COUNT: int = int(os.getenv("RABBITMQ_PREFETCH_COUNT", "1"))
    RABBITMQ_MAX_RETRIES: int = int(os.getenv("RABBITMQ_MAX_RETRIES", "5"))
    RABBITMQ_RETRY_DELAY_MS: int = int(os.getenv("RABBITMQ_RETRY_DELAY_MS", "300000"))
    RABBITMQ_FAILED_TTL_MS: int = int(os.getenv("RABBITMQ_FAILED_TTL_MS", "3600000"))
    MAX_DEQUEUE_CONCURRENCY: int = int(os.getenv("MAX_DEQUEUE_CONCURRENCY", "1"))
    UNSTRUCTURED_FANOUT_CONCURRENCY: int = int(os.getenv("UNSTRUCTURED_FANOUT_CONCURRENCY", "10"))
    NEO4J_URI: str | None = os.getenv("NEO4J_URI")
    NEO4J_USERNAME: str | None = os.getenv("NEO4J_USERNAME")
    NEO4J_PASSWORD: str | None = os.getenv("NEO4J_PASSWORD")
    NEO4J_DATABASE: str | None = os.getenv("NEO4J_DATABASE")
    PDF_IMAGE_DPI: int = int(os.getenv("PDF_IMAGE_DPI", "100"))
    NEO4J_MAX_TRANSACTION_RETRIES: int = int(os.getenv("NEO4J_MAX_TRANSACTION_RETRIES", "5"))
    NEO4J_RETRY_BACKOFF_SECONDS: float = float(os.getenv("NEO4J_RETRY_BACKOFF_SECONDS", "0.5"))

settings = Settings()
