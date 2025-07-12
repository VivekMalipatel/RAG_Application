import os

class Config:

    # Models Configuration
    REASONING_LLM_MODEL: str = "Qwen/Qwen3-8B-AWQ"
    VLM_MODEL: str = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
    MULTIMODEL_EMBEDDING_MODEL: str = "VivekMalipatel23/nomic-embed-multimodal-3b-8bit"
    MULTIMODEL_EMBEDDING_MODEL_DIMS: int = 2048
    TEXT_EMBEDDING_MODEL: str = "nomic-ai/nomic-embed-text-v1.5"
    TEXT_EMBEDDING_MODEL_DIMS: int = 768
    OPENAI_BASE_URL: str = "https://llm.gauravshivaprasad.com/v2"
    OPENAI_API_KEY: str = "sk-372c69b72fb14a90a2e1b0b17884d9b4"
    MODEL_PROVIDER: str = "openai"

    # Redis Configuration
    REDIS_HOST: str = "192.168.0.20"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str = "password"
    REDIS_DB: int = 0
    
    #Prompt Configuration
    MEDIA_DESCRIPTION_PROMPT: str = "Provide an extremely detailed description of this media content. Include every visible/audible element, text, object, person, color, layout, sounds, speech, and any other relevant details without missing anything."

    # State Management
    MAX_STATE_TOKENS: int = 40000

config = Config()