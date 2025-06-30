import os
import re
import json
import logging
from typing import Dict, List, Optional, Union
from pydantic_settings import BaseSettings
from pydantic import model_validator
import secrets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

class Settings(BaseSettings):
    API_V1_STR: str = os.getenv("API_V1_STR", "/v1")
    PROJECT_NAME: str = os.getenv("PROJECT_NAME", "ModelRouter API")
    
    SECRET_KEY: str = os.getenv("SECRET_KEY", secrets.token_urlsafe(32))
    API_KEY_HEADER: str = os.getenv("API_KEYS", "test-key")
    BEARER_TOKEN_HEADER: str = os.getenv("BEARER_TOKEN_HEADER", "Authorization")
    API_KEYS: List[str] = json.loads(os.getenv("API_KEYS", '["test-key"]'))
    USE_BEARER_TOKEN: bool = os.getenv("USE_BEARER_TOKEN", "True").lower() in ("true", "1", "yes")
    
    DATABASE_URL: str = "sqlite:///./modelrouter.db"
    
    HUGGINGFACE_API_TOKEN: Optional[str] = None
    
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_NUM_CTX: int = 8096

    OPENAI_DEFAULT_SYSTEM_PROMPT: Optional[str] = "You are a helpful AI assistant."
    OPENAI_DEFAULT_TEMPERATURE: float = 0.7
    OPENAI_DEFAULT_TOP_P: float = 0.9
    OPENAI_DEFAULT_STREAM: bool = False
    OPENAI_DEFAULT_FREQUENCY_PENALTY: float = 0.0
    OPENAI_DEFAULT_PRESENCE_PENALTY: float = 0.0
    OPENAI_DEFAULT_N: int = 1
    OPENAI_DEFAULT_PARALLEL_TOOL_CALLS: bool = True
    OPENAI_DEFAULT_LOGPROBS: bool = False
    
    OLLAMA_DEFAULT_TEMPERATURE: float = 0.7
    OLLAMA_DEFAULT_TOP_P: float = 0.9
    OLLAMA_DEFAULT_STREAM: bool = False
    OLLAMA_DEFAULT_FREQUENCY_PENALTY: float = 0.0
    OLLAMA_DEFAULT_PRESENCE_PENALTY: float = 0.0
    OLLAMA_DEFAULT_N: int = 1
    OLLAMA_DEFAULT_PARALLEL_TOOL_CALLS: bool = True
    OLLAMA_DEFAULT_NUM_CTX: int = 8192
    OLLAMA_DEFAULT_REPEAT_LAST_N: int = 128
    OLLAMA_DEFAULT_REPEAT_PENALTY: float = 1.1
    OLLAMA_DEFAULT_TOP_K: int = 40
    OLLAMA_DEFAULT_MIN_P: float = 0.05
    OLLAMA_DEFAULT_KEEP_ALIVE: str = "10m"
    OLLAMA_DEFAULT_THINK: bool = False
    OLLAMA_DEFAULT_MAX_RETRIES: int = 3
    OLLAMA_DEFAULT_RETRY_DELAY: float = 1.0
    OLLAMA_DEFAULT_CONNECTION_TIMEOUT: float = 60.0
    
    CHAT_DEFAULT_FREQUENCY_PENALTY: float = 0.0
    CHAT_DEFAULT_LOGPROBS: bool = False
    CHAT_DEFAULT_N: int = 1
    CHAT_DEFAULT_PRESENCE_PENALTY: float = 0.0
    CHAT_DEFAULT_STREAM: bool = False
    CHAT_DEFAULT_TEMPERATURE: float = 0.7
    CHAT_DEFAULT_TOP_P: float = 0.9
    CHAT_DEFAULT_PARALLEL_TOOL_CALLS: bool = True
    CHAT_DEFAULT_RESPONSE_FORMAT_TYPE: Optional[str] = None
    CHAT_DEFAULT_SERVICE_TIER: Optional[str] = None
    CHAT_DEFAULT_OBJECT_COMPLETION: str = "chat.completion"
    CHAT_DEFAULT_OBJECT_CHUNK: str = "chat.completion.chunk"

    TEXT_GENERATION_MODELS: List[str] = ["gpt-4o-mini","gpt-4o","qwen2.5-vl-72b-instruct","deepseek-chat","deepseek-reasoner","qwen2.5:7b-instruct-q8_0","qwen2.5vl:3b-q4_K_M","qwen2.5vl:7b-q8_0","qwen3:8b-q8_0","gemma3:12b-it-q8_0"]
    TEXT_EMBEDDING_MODELS: List[str] = ["nomic-ai/nomic-embed-multimodal-3b"]
    IMAGE_EMBEDDING_MODELS: List[str] = ["nomic-ai/nomic-embed-multimodal-3b"]
    RERANKER_MODELS: List[str] = ["jinaai/jina-colbert-v2"]
    
    HUGGINGFACE_MODELS: List[str] = ["nomic-ai/nomic-embed-multimodal-3b", "jinaai/jina-colbert-v2"]
    OLLAMA_MODELS: List[str] = ["qwen2.5:7b-instruct-q8_0","qwen2.5vl:3b-q4_K_M","qwen2.5vl:7b-q8_0","qwen3:8b-q8_0","gemma3:12b-it-q8_0"]
    
    OPENAI_ORGS: List[str] = ["OPENAI", "DEEPSEEK", "QWEN"]

    openai_compatible_providers: Dict[str, Dict[str, Union[str, List[str]]]] = {}

    @model_validator(mode='after')
    def load_openai_compatible_providers(self):
        providers = {}
        
        for provider_name in self.OPENAI_ORGS:
            provider_name = provider_name.strip()
            if not provider_name:
                continue
                
            models_var = f"{provider_name}_MODELS"
            api_key_var = f"{provider_name}_API_KEY"
            base_url_var = f"{provider_name}_BASE_URL"
            
            models_env = os.getenv(models_var)
            if not models_env:
                continue
                
            api_key = os.getenv(api_key_var)
            base_url = os.getenv(base_url_var)
            
            if not api_key:
                raise ValueError(f"Missing required environment variable: {api_key_var}")
            if not base_url:
                raise ValueError(f"Missing required environment variable: {base_url_var}")
            
            try:
                models = json.loads(models_env)
                if not isinstance(models, list):
                    raise ValueError(f"Expected list format for {models_var}")
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON format for {models_var}: {models_env}")
            
            providers[provider_name] = {
                'api_key': api_key,
                'base_url': base_url,
                'models': models
            }
        
        self.openai_compatible_providers = providers
        
        logging.info(f"Loaded {len(providers)} OpenAI-compatible providers:")
        for provider_name, config in providers.items():
            model_count = len(config['models'])
            models_list = ', '.join(config['models'])
            logging.info(f"  {provider_name}: {model_count} models - {models_list}")
        
        return self

    def get_provider_config(self, model_name: str) -> Optional[Dict[str, str]]:
        for provider_name, config in self.openai_compatible_providers.items():
            if model_name in config['models']:
                return {
                    'provider_name': provider_name,
                    'api_key': config['api_key'],
                    'base_url': config['base_url']
                }
        return None

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "allow"

settings = Settings()