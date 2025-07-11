import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    API_TITLE: str = "File Indexer API"
    API_DESCRIPTION: str = "API for indexing various file types for RAG applications"
    API_VERSION: str = "0.1.0"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    DB_URL: str = os.getenv("DB_URL")

    UNOSERVER_HOST: str = os.getenv("UNOSERVER_HOST")
    UNOSERVER_PORT: int = os.getenv("UNOSERVER_PORT")

    INFERENCE_API_KEY: str = os.getenv("INFERENCE_API_KEY")
    INFERENCE_API_BASE: str = os.getenv("INFERENCE_API_BASE")
    EMBEDDING_API_KEY: str = os.getenv("EMBEDDING_API_KEY")
    EMBEDDING_API_BASE: str = os.getenv("EMBEDDING_API_BASE")

    MINIO_ENDPOINT_URL: str = os.getenv("MINIO_ENDPOINT_URL")
    S3_BUCKET_NAME: str = os.getenv("S3_BUCKET_NAME")
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_REGION: str = os.getenv("AWS_REGION")

    AUTO_BACKUP_INTERVAL_MINUTES: int = int(os.getenv("AUTO_BACKUP_INTERVAL_MINUTES", "30"))
    S3_BACKUP_PREFIX: str = os.getenv("S3_BACKUP_PREFIX", "IndexerAPI/")

    # RabbitMQ Configuration
    RABBITMQ_URL: str = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
    RABBITMQ_QUEUE_NAME: str = os.getenv("RABBITMQ_QUEUE_NAME", "indexer_queue")
    RABBITMQ_EXCHANGE_NAME: str = os.getenv("RABBITMQ_EXCHANGE_NAME", "indexer_exchange")
    RABBITMQ_ROUTING_KEY: str = os.getenv("RABBITMQ_ROUTING_KEY", "indexer.task")

settings = Settings()