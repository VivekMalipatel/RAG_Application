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

    # Model Configuration
    INFERENCE_API_KEY: str = os.getenv("INFERENCE_API_KEY")
    INFERENCE_API_BASE: str = os.getenv("INFERENCE_API_BASE")
    EMBEDDING_API_KEY: str = os.getenv("EMBEDDING_API_KEY")
    EMBEDDING_API_BASE: str = os.getenv("EMBEDDING_API_BASE")

    INFERENCE_MODEL: str = os.getenv("INFERENCE_MODEL")
    REASONING_MODEL: str = os.getenv("REASONING_MODEL")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL")
    EMBEDDING_DIMENSIONS: int = int(os.getenv("EMBEDDING_DIMENSIONS", "2048"))
    STRUCTURED_OUTPUTS_MAX_TOKENS: int = int(os.getenv("ENTITY_RELATION_EXTRACTION_MAX_TOKENS", "110000"))


    EMBEDDING_CONCURRENT_REQUESTS: int = int(os.getenv("EMBEDDING_CONCURRENT_REQUESTS", "16"))
    INFERENCE_CONCURRENT_REQUESTS: int = int(os.getenv("INFERENCE_CONCURRENT_REQUESTS", "16"))
    EMBEDDING_CLIENT_TIMEOUT: float = float(os.getenv("EMBEDDING_CLIENT_TIMEOUT", "120"))
    INFERENCE_CLIENT_TIMEOUT: float = float(os.getenv("INFERENCE_CLIENT_TIMEOUT", "120"))
    RETRIES: int = int(os.getenv("RETRIES", "3"))
    RETRY_DELAY: int = int(os.getenv("RETRY_DELAY", "2"))


    MINIO_ENDPOINT_URL: str = os.getenv("MINIO_ENDPOINT_URL")
    S3_BUCKET_NAME: str = os.getenv("S3_BUCKET_NAME")
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_REGION: str = os.getenv("AWS_REGION")

    # RabbitMQ Configuration
    RABBITMQ_URL: str = os.getenv("RABBITMQ_URL")
    RABBITMQ_QUEUE_NAME: str = os.getenv("RABBITMQ_QUEUE_NAME")
    RABBITMQ_EXCHANGE_NAME: str = os.getenv("RABBITMQ_EXCHANGE_NAME")
    RABBITMQ_ROUTING_KEY: str = os.getenv("RABBITMQ_ROUTING_KEY")
    RABBITMQ_HEARTBEAT: int = int(os.getenv("RABBIT_MQ_HEARTBEAT", "30"))
    RABBITMQ_CONSUMER_TIMEOUT: int = int(os.getenv("RABBIT_MQ_CONSUMER_TIMEOUT", "30"))
    RABBITMQ_X_CONSUMER_TIMEOUT: int = int(os.getenv("RABBIT_MQ_X_CONSUMER_TIMEOUT", "36000000"))
    
    # Consumer Configuration
    MAX_DEQUEUE_CONCURRENCY: int = int(os.getenv("MAX_DEQUEUE_CONCURRENCY", "5"))

    # Neo4j Configuration
    NEO4J_URI: str = os.getenv("NEO4J_URI")
    NEO4J_USERNAME: str = os.getenv("NEO4J_USERNAME")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD")
    NEO4J_DATABASE: str = os.getenv("NEO4J_DATABASE")

    # Processing Configuration
    PDF_IMAGE_DPI: int = int(os.getenv("PDF_IMAGE_DPI", "100"))

settings = Settings()