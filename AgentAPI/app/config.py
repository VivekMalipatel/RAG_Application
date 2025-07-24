import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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
    LLM_TIMEOUT: float = float(os.getenv("LLM_TIMEOUT", 240))
    LLM_MAX_RETRIES: int = int(os.getenv("REASONING_LLM_MAX_RETRIES", 3))
    REASONING_LLM_MAX_TOKENS: int = int(os.getenv("REASONING_LLM_MAX_TOKENS", 50000))
    VLM_LLM_MAX_TOKENS: int = int(os.getenv("VLM_LLM_MAX_TOKENS", 120000))

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

    # Searx Configuration
    SEARX_URL: str = "https://websearch.gauravshivaprasad.com"
    # State Management
    MAX_STATE_TOKENS: int = int(os.getenv("MAX_STATE_TOKENS", 20000))
    
    # MCP Server Configuration
    MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://10.9.0.5:8082/mcp")
    MCP_JSON_PATH = os.getenv(
        "MCP_JSON_PATH",
        os.path.join(BASE_DIR, "tools/core_tools/mcp/mcp.json")
    )


config = Config()