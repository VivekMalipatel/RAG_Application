import logging
import asyncio
import time
from typing import List, Dict, Any, Optional, Union, AsyncGenerator
import httpx

from config import settings
from model_type import ModelType

class HuggingFaceClient:
    """
    Client for interacting with Hugging Face's API.
    Supports inference API for completions, embeddings, and other tasks.
    """
    
    def __init__(
        self,
        model_name: str,
        model_type: ModelType,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ):
        """
        Initializes the Hugging Face client.
        
        Args:
            model_name: Full model name (e.g., "mistralai/Mistral-7B-Instruct-v0.1")
            model_type: Type of model (text generation, embedding, etc.)
            system_prompt: System prompt for chat models
            temperature: Temperature parameter for sampling
            top_p: Nucleus sampling parameter
            max_tokens: Maximum tokens to generate
            stream: Whether to stream responses
        """
        self.model_name = model_name
        self.model_type = model_type
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.stream = stream
        
        # Extract additional parameters
        self.repetition_penalty = kwargs.get("repetition_penalty", 1.0)
        self.top_k = kwargs.get("top_k", 50)
        self.num_beams = kwargs.get("num_beams", 1)
        
        # HF API settings
        self.api_url = "https://api-inference.huggingface.co/models"
        self.api_token = settings.HUGGINGFACE_API_TOKEN
        self.headers = {"Authorization": f"Bearer {self.api_token}"}
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized Hugging Face client with model {model_name}")
    
    def is_model_available(self) -> bool:
        """Check if the model is available in Hugging Face."""
        try:
            # Check if we have an API token first
            if not self.api_token:
                self.logger.warning("No Hugging Face API token provided")
                return False
                
            # Simple synchronous check
            with httpx.Client(timeout=10.0) as client:
                response = client.get(
                    f"{self.api_url}/{self.model_name}", 
                    headers=self.headers
                )
                return response.status_code == 200
                
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
        Generate text using Hugging Face's API.
        
        Args:
            prompt: Text prompt to generate response for
            max_tokens: Maximum number of tokens to generate
            stream: Whether to enable streaming
            
        Returns:
            Either the generated text or an AsyncGenerator yielding text chunks
        """
        stream = stream if stream is not None else self.stream
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        # If using a model that expects chat-like formatting, apply it
        full_prompt = self._format_chat_prompt(prompt)
        
        # Build request payload
        payload = {
            "inputs": full_prompt,
            "parameters": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "repetition_penalty": self.repetition_penalty,
                "do_sample": True,
                "num_beams": self.num_beams,
            }
        }
        
        if max_tokens:
            payload["parameters"]["max_new_tokens"] = max_tokens
        
        # Streaming not directly supported in HF API, we simulate it
        if stream:
            return self._simulate_streaming(full_prompt, payload)
        
        # Handle non-streaming response
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.post(
                    f"{self.api_url}/{self.model_name}",
                    headers=self.headers,
                    json=payload,
                    timeout=60.0
                )
                
                if response.status_code != 200:
                    error_detail = response.text
                    self.logger.error(f"Hugging Face API error: {response.status_code} - {error_detail}")
                    return f"Error: {response.status_code} - {error_detail}"
                
                # Extract generated text
                response_json = response.json()
                
                if isinstance(response_json, list) and len(response_json) > 0:
                    if isinstance(response_json[0], dict) and "generated_text" in response_json[0]:
                        generated_text = response_json[0]["generated_text"]
                        
                        # Remove the prompt from the generated text
                        if generated_text.startswith(full_prompt):
                            generated_text = generated_text[len(full_prompt):]
                            
                        return generated_text.strip()
                    else:
                        return str(response_json[0])
                else:
                    return str(response_json)
                    
            except Exception as e:
                self.logger.error(f"Error with Hugging Face API: {str(e)}")
                return f"Error: {str(e)}"
    
    async def _simulate_streaming(
        self, 
        prompt: str, 
        payload: Dict[str, Any]
    ) -> AsyncGenerator[str, None]:
        """
        Simulate streaming for Hugging Face models by chunking the response.
        
        Args:
            prompt: The full prompt including system instructions
            payload: The request payload
            
        Yields:
            Text chunks as they would be streamed
        """
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.post(
                    f"{self.api_url}/{self.model_name}",
                    headers=self.headers,
                    json=payload,
                    timeout=60.0
                )
                
                if response.status_code != 200:
                    error_detail = response.text
                    self.logger.error(f"Hugging Face API error: {response.status_code} - {error_detail}")
                    yield f"Error: {response.status_code} - {error_detail}"
                    return
                
                # Extract generated text
                response_json = response.json()
                
                if isinstance(response_json, list) and len(response_json) > 0:
                    if isinstance(response_json[0], dict) and "generated_text" in response_json[0]:
                        generated_text = response_json[0]["generated_text"]
                        
                        # Remove the prompt from the generated text
                        if generated_text.startswith(prompt):
                            generated_text = generated_text[len(prompt):]
                            
                        generated_text = generated_text.strip()
                        
                        # Simulate streaming by yielding chunks of the text
                        words = generated_text.split()
                        chunk_size = max(1, len(words) // 10)  # Aim for ~10 chunks
                        
                        for i in range(0, len(words), chunk_size):
                            chunk = " ".join(words[i:i+chunk_size])
                            yield chunk + " "
                            await asyncio.sleep(0.1)  # Simulate delay between chunks
                    else:
                        yield str(response_json[0])
                else:
                    yield str(response_json)
                    
            except Exception as e:
                self.logger.error(f"Error with Hugging Face API: {str(e)}")
                yield f"Error: {str(e)}"
    
    async def embed_text(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for text using Hugging Face's API.
        
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
                        "inputs": text,
                        "options": {"wait_for_model": True}
                    }
                    
                    response = await client.post(
                        f"{self.api_url}/{self.model_name}",
                        headers=self.headers,
                        json=payload,
                        timeout=30.0
                    )
                    
                    if response.status_code != 200:
                        error_detail = response.text
                        self.logger.error(f"Hugging Face API error: {response.status_code} - {error_detail}")
                        # Append zeros as fallback
                        embeddings.append([0.0] * 768)  # Common embedding size
                        continue
                    
                    # Parse the response
                    response_json = response.json()
                    
                    if isinstance(response_json, list) and len(response_json) > 0:
                        # Handle different response formats from different models
                        if isinstance(response_json[0], list):
                            # Some models return a list of lists directly
                            embeddings.append(response_json[0])
                        else:
                            # Others return a list of values
                            embeddings.append(response_json)
                    else:
                        # Yet others might return a dictionary with embeddings
                        if isinstance(response_json, dict) and "embedding" in response_json:
                            embeddings.append(response_json["embedding"])
                        else:
                            self.logger.warning(f"Unexpected embedding response format: {response_json}")
                            embeddings.append([0.0] * 768)  # Common embedding size
                            
                except Exception as e:
                    self.logger.error(f"Error getting embeddings from Hugging Face: {str(e)}")
                    embeddings.append([0.0] * 768)  # Common embedding size
        
        return embeddings
    
    def _format_chat_prompt(self, prompt: str) -> str:
        """
        Format the prompt according to the expected format for the model.
        
        Args:
            prompt: The raw user prompt
            
        Returns:
            A formatted prompt with system message if applicable
        """
        # Check if model is a known chat model that requires special formatting
        model_name_lower = self.model_name.lower()
        
        # Mistral-specific formatting
        if "mistral" in model_name_lower and "instruct" in model_name_lower:
            if self.system_prompt:
                return f"<s>[INST] {self.system_prompt}\n\n{prompt} [/INST]"
            else:
                return f"<s>[INST] {prompt} [/INST]"
                
        # Llama2-specific formatting
        elif "llama" in model_name_lower and "chat" in model_name_lower:
            if self.system_prompt:
                return f"<s>[INST] <<SYS>>\n{self.system_prompt}\n<</SYS>>\n\n{prompt} [/INST]"
            else:
                return f"<s>[INST] {prompt} [/INST]"
                
        # Default: just combine system prompt and user prompt with a newline
        else:
            if self.system_prompt:
                return f"{self.system_prompt}\n\n{prompt}"
            else:
                return prompt
                
    def set_system_prompt(self, system_prompt: str):
        """Update the system prompt."""
        self.system_prompt = system_prompt