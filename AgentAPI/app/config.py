import os

class Config:

    # Models Configuration
    REASONING_LLM_MODEL: str = os.getenv("REASONING_LLM_MODEL")
    VLM_MODEL: str = os.getenv("VLM_MODEL")
    MULTIMODEL_EMBEDDING_MODEL: str = os.getenv("MULTIMODEL_EMBEDDING_MODEL")
    MULTIMODEL_EMBEDDING_MODEL_DIMS: int = int(os.getenv("MULTIMODEL_EMBEDDING_MODEL_DIMS",2048))
    TEXT_EMBEDDING_MODEL: str = os.getenv("TEXT_EMBEDDING_MODEL")
    TEXT_EMBEDDING_MODEL_DIMS: int = int(os.getenv("TEXT_EMBEDDING_MODEL_DIMS", 768))
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    MODEL_PROVIDER: str = os.getenv("MODEL_PROVIDER")

    INDEXER_API_BASE_URL: str = os.getenv("INDEXER_API_BASE_URL")

    # Redis Configuration
    REDIS_HOST: str = os.getenv("REDIS_HOST")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT"))
    REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD")
    REDIS_DB: int = int(os.getenv("REDIS_DB"))

    # Prompt Configuration
    MEDIA_DESCRIPTION_PROMPT: str = os.getenv(
        "MEDIA_DESCRIPTION_PROMPT"
    )

    # State Management
    MAX_STATE_TOKENS: int = int(os.getenv("MAX_STATE_TOKENS", 40000))

config = Config()