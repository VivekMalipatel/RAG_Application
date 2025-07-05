import os

class Config:
    REASONING_LLM_MODEL: str = "Qwen/Qwen3-8B-AWQ"
    VLM_MODEL: str = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
    OPENAI_BASE_URL: str = "https://llm.gauravshivaprasad.com/v2"
    OPENAI_API_KEY: str = "sk-372c69b72fb14a90a2e1b0b17884d9b4"
    MODEL_PROVIDER: str = "openai"
    
    MEDIA_DESCRIPTION_PROMPT: str = "Provide an extremely detailed description of this media content. Include every visible/audible element, text, object, person, color, layout, sounds, speech, and any other relevant details without missing anything."

config = Config()