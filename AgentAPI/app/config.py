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
    LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", 3))
    EMBEDDING_CLIENT_TIMEOUT: float = float(os.getenv("EMBEDDING_CLIENT_TIMEOUT", 30))
    EMBEDDING_CLEINT_RETRIES: int = int(os.getenv("EMBEDDING_CLEINT_RETRIES", 1))

    REASONING_LLM_TEMPERATURE: float = float(os.getenv("REASONING_LLM_TEMPERATURE", 0.6))
    REASONING_LLM_TOP_P: float = float(os.getenv("REASONING_LLM_TOP_P", 0.95))
    REASONING_LLM_REPETITION_PENALTY: float = float(os.getenv("REASONING_LLM_REPETITION_PENALTY", 1.05))
    REASONING_LLM_PRESENCE_PENALTY: float = float(os.getenv("REASONING_LLM_PRESENCE_PENALTY", 1.25))
    REASONING_LLM_TOP_K: int = int(os.getenv("REASONING_LLM_TOP_K", 20))
    REASONING_LLM_MIN_P: int = int(os.getenv("REASONING_LLM_MIN_P", 0))

    VLM_LLM_TEMPERATURE: float = float(os.getenv("VLM_LLM_TEMPERATURE", 0.1))
    VLM_LLM_TOP_P: float = float(os.getenv("VLM_LLM_TOP_P", 0.95))
    VLM_LLM_REPETITION_PENALTY: float = float(os.getenv("VLM_LLM_REPETITION_PENALTY", 1.05))
    VLM_LLM_PRESENCE_PENALTY: float = float(os.getenv("VLM_LLM_PRESENCE_PENALTY", 1.5))
    VLM_LLM_TOP_K: int = int(os.getenv("REASONING_LLM_TOP_K", 20))
    VLM_LLM_MIN_P: int = int(os.getenv("REASONING_LLM_MIN_P", 0))

    SEMANTIC_MEMORY_DELAY_SECONDS: int = int(os.getenv("SEMANTIC_MEMORY_DELAY_SECONDS", 60))
    SEMANTIC_MEMORY_QUERY_LIMIT: int = int(os.getenv("SEMANTIC_MEMORY_QUERY_LIMIT", 10))
    SEMANTIC_MEMORY_MAX_UPDATES_PER_TURN: int = int(os.getenv("SEMANTIC_MEMORY_MAX_UPDATES_PER_TURN", 1))
    PROFILE_MEMORY_DELAY_SECONDS: int = int(os.getenv("PROFILE_MEMORY_DELAY_SECONDS", 60))
    PROFILE_MEMORY_MAX_UPDATES_PER_TURN: int = int(os.getenv("PROFILE_MEMORY_MAX_UPDATES_PER_TURN", 1))
    PROFILE_MEMORY_MIN_CONFIDENCE: float = float(os.getenv("PROFILE_MEMORY_MIN_CONFIDENCE", 0.5))
    PROFILE_MEMORY_QUERY_LIMIT: int = int(os.getenv("PROFILE_MEMORY_QUERY_LIMIT", 10))
    PROCEDURAL_MEMORY_DELAY_SECONDS: int = int(os.getenv("PROCEDURAL_MEMORY_DELAY_SECONDS", 60))
    PROCEDURAL_MEMORY_MAX_UPDATES_PER_TURN: int = int(os.getenv("PROCEDURAL_MEMORY_MAX_UPDATES_PER_TURN", 1))
    PROCEDURAL_MEMORY_QUERY_LIMIT: int = int(os.getenv("PROCEDURAL_MEMORY_QUERY_LIMIT", 5))
    EPISODIC_MEMORY_DELAY_SECONDS: int = int(os.getenv("EPISODIC_MEMORY_DELAY_SECONDS", 60))
    EPISODIC_MEMORY_MAX_UPDATES_PER_TURN: int = int(os.getenv("EPISODIC_MEMORY_MAX_UPDATES_PER_TURN", 1))
    EPISODIC_MEMORY_QUERY_LIMIT: int = int(os.getenv("EPISODIC_MEMORY_QUERY_LIMIT", 5))

    INDEXER_API_BASE_URL: str = os.getenv("INDEXER_API_BASE_URL")

    # Redis Configuration
    REDIS_HOST: str = os.getenv("REDIS_HOST")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT"))
    REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD")
    REDIS_DB: int = int(os.getenv("REDIS_DB"))

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