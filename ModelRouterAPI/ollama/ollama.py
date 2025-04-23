import logging
import json
import asyncio
from typing import List, Dict, Any, Optional, Union, AsyncGenerator
import httpx

from config import settings

class OllamaClient:
    """
    Client for interacting with Ollama's API.
    Supports local model inference for completions and embeddings.
    """
    
    def __init__(
        self,
        hf_repo: str,
        system_prompt: Optional[str] = None,
        quantization: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ):
        """
        Initializes the Ollama client.
        
        Args:
            hf_repo: Model identifier (e.g., "llama2", "mistral")
            system_prompt: System prompt for chat models
            quantization: Quantization level (if applicable)
            temperature: Temperature parameter for sampling
            top_p: Nucleus sampling parameter
            max_tokens: Maximum tokens to generate
            stream: Whether to stream responses
        """
        self.model_name = hf_repo
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.stream = stream
        
        # Extract additional parameters
        self.top_k = kwargs.get("top_k", 40)
        self.repeat_penalty = kwargs.get("repeat_penalty", 1.1)
        
        self.api_base_url = settings.OLLAMA_BASE_URL
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized Ollama client with model {hf_repo}")
    
    def is_model_available(self) -> bool:
        """Check if the model is available in Ollama."""
        try:
            # Simple synchronous check
            with httpx.Client(timeout=10.0) as client:
                response = client.get(f"{self.api_base_url}/api/tags")
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    model_names = [model.get("name", "") for model in models]
                    return self.model_name in model_names
                return False
        except Exception as e:
            self.logger.error(f"Error checking model availability: {str(e)}")
            return False
    
    async def generate_text(
        self, 
        prompt: str,
        max_tokens: Optional[int] = None,
        stream: Optional[bool] = None
    ) -> Union[str, AsyncGenerator[str, None]]:
        """
        Generate text using Ollama's API.
        
        Args:
            prompt: Text prompt to generate response for
            max_tokens: Maximum number of tokens to generate
            stream: Whether to enable streaming
            
        Returns:
            Either the generated text or an AsyncGenerator yielding text chunks
        """
        stream = stream if stream is not None else self.stream
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        # Build request payload
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "repeat_penalty": self.repeat_penalty
            }
        }
        
        if self.system_prompt:
            payload["system"] = self.system_prompt
            
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                # Handle streaming response
                if stream:
                    async def generate_stream():
                        async with client.stream(
                            "POST", 
                            f"{self.api_base_url}/api/generate",
                            json=payload, 
                            timeout=60.0
                        ) as response:
                            if response.status_code != 200:
                                error_detail = await response.text()
                                self.logger.error(f"Ollama API error: {response.status_code} - {error_detail}")
                                yield f"Error: {response.status_code} - {error_detail}"
                                return
                                
                            # Stream the response as it comes
                            async for chunk in response.aiter_lines():
                                if not chunk:
                                    continue
                                try:
                                    chunk_data = json.loads(chunk)
                                    if "response" in chunk_data:
                                        yield chunk_data["response"]
                                except json.JSONDecodeError:
                                    yield chunk  # In case it's not valid JSON
                    
                    return generate_stream()
                
                # Handle non-streaming response
                else:
                    response = await client.post(
                        f"{self.api_base_url}/api/generate",
                        json=payload,
                        timeout=60.0
                    )
                    
                    if response.status_code != 200:
                        error_detail = response.text
                        self.logger.error(f"Ollama API error: {response.status_code} - {error_detail}")
                        return f"Error: {response.status_code} - {error_detail}"
                        
                    response_data = response.json()
                    return response_data.get("response", "")
                    
            except Exception as e:
                self.logger.error(f"Error with Ollama API: {str(e)}")
                if stream:
                    # Return an empty generator for streaming
                    async def error_generator():
                        yield f"Error: {str(e)}"
                    return error_generator()
                else:
                    return f"Error: {str(e)}"
    
    async def embed_text(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for text using Ollama's API.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            for text in texts:
                try:
                    payload = {
                        "model": self.model_name,
                        "prompt": text,
                    }
                    
                    response = await client.post(
                        f"{self.api_base_url}/api/embeddings",
                        json=payload,
                        timeout=30.0
                    )
                    
                    if response.status_code != 200:
                        error_detail = response.text
                        self.logger.error(f"Ollama API error: {response.status_code} - {error_detail}")
                        # Append zeros as fallback
                        embeddings.append([0.0] * 768)  # Common embedding size
                        continue
                        
                    response_data = response.json()
                    embeddings.append(response_data.get("embedding", []))
                    
                except Exception as e:
                    self.logger.error(f"Error getting embeddings from Ollama: {str(e)}")
                    # Append zeros as fallback
                    embeddings.append([0.0] * 768)  # Common embedding size
        
        return embeddings
        
    def set_system_prompt(self, system_prompt: str):
        """Update the system prompt."""
        self.system_prompt = system_prompt