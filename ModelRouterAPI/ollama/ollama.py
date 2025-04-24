import logging
import json
import asyncio
from typing import List, Dict, Any, Optional, Union, AsyncGenerator, Type
import httpx
import time
from pydantic import BaseModel

from config import settings

class OllamaClient:
    """
    Client for interacting with Ollama's API.
    Supports local model inference for completions and embeddings.
    Includes connection retry logic for remote Ollama servers.
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
        
        # Connection settings
        self.api_base_url = settings.OLLAMA_BASE_URL
        self.max_retries = kwargs.get("max_retries", 3)
        self.retry_delay = kwargs.get("retry_delay", 1.0)  # seconds
        self.connection_timeout = kwargs.get("connection_timeout", 10.0)  # seconds
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized Ollama client with model {hf_repo} at {self.api_base_url}")
        
        # Check if the Ollama server is available
        if not self.check_server_health():
            self.logger.warning(f"Ollama server at {self.api_base_url} is not responding. Some operations may fail.")
    
    def check_server_health(self) -> bool:
        """Check if the Ollama server is running and responding."""
        try:
            with httpx.Client(timeout=self.connection_timeout) as client:
                response = client.get(f"{self.api_base_url}/api/version")
                if response.status_code == 200:
                    version = response.json().get("version", "unknown")
                    self.logger.info(f"Connected to Ollama server version {version}")
                    return True
                self.logger.error(f"Ollama server returned status code {response.status_code}")
                return False
        except Exception as e:
            self.logger.error(f"Failed to connect to Ollama server: {str(e)}")
            return False
    
    def is_model_available(self) -> bool:
        """Check if the model is available in Ollama."""
        for attempt in range(self.max_retries):
            try:
                # Simple synchronous check
                with httpx.Client(timeout=self.connection_timeout) as client:
                    response = client.get(f"{self.api_base_url}/api/tags")
                    if response.status_code == 200:
                        models = response.json().get("models", [])
                        model_names = [model.get("name", "") for model in models]
                        return self.model_name in model_names
                    self.logger.warning(f"Failed to check model availability, status: {response.status_code}")
                    return False
            except httpx.ConnectTimeout:
                self.logger.warning(f"Connection timeout checking model availability (attempt {attempt+1}/{self.max_retries})")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                continue
            except Exception as e:
                self.logger.error(f"Error checking model availability: {str(e)}")
                return False
        
        self.logger.error(f"Failed to check model availability after {self.max_retries} attempts")
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
        
        # Setup httpx client with retry capability
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Handle streaming response
            if stream:
                async def generate_stream():
                    for attempt in range(self.max_retries):
                        try:
                            async with client.stream(
                                "POST", 
                                f"{self.api_base_url}/api/generate",
                                json=payload, 
                                timeout=60.0
                            ) as response:
                                if response.status_code != 200:
                                    error_detail = await response.text()
                                    self.logger.error(f"Ollama API error: {response.status_code} - {error_detail}")
                                    if attempt < self.max_retries - 1:
                                        self.logger.info(f"Retrying... (attempt {attempt+1}/{self.max_retries})")
                                        await asyncio.sleep(self.retry_delay)
                                        continue
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
                                
                                # If we got here, the request succeeded
                                break
                                
                        except (httpx.ConnectTimeout, httpx.ReadTimeout) as e:
                            self.logger.warning(f"Timeout during streaming request: {str(e)}")
                            if attempt < self.max_retries - 1:
                                self.logger.info(f"Retrying... (attempt {attempt+1}/{self.max_retries})")
                                await asyncio.sleep(self.retry_delay)
                                continue
                            yield f"Error: Connection timeout after {self.max_retries} attempts"
                            return
                        except Exception as e:
                            self.logger.error(f"Error during streaming request: {str(e)}")
                            yield f"Error: {str(e)}"
                            return
                
                return generate_stream()
            
            # Handle non-streaming response
            else:
                for attempt in range(self.max_retries):
                    try:
                        response = await client.post(
                            f"{self.api_base_url}/api/generate",
                            json=payload,
                            timeout=60.0
                        )
                        
                        if response.status_code != 200:
                            error_detail = response.text
                            self.logger.error(f"Ollama API error: {response.status_code} - {error_detail}")
                            if attempt < self.max_retries - 1:
                                self.logger.info(f"Retrying... (attempt {attempt+1}/{self.max_retries})")
                                await asyncio.sleep(self.retry_delay)
                                continue
                            return f"Error: {response.status_code} - {error_detail}"
                            
                        response_data = response.json()
                        return response_data.get("response", "")
                        
                    except (httpx.ConnectTimeout, httpx.ReadTimeout) as e:
                        self.logger.warning(f"Timeout error: {str(e)}")
                        if attempt < self.max_retries - 1:
                            self.logger.info(f"Retrying... (attempt {attempt+1}/{self.max_retries})")
                            await asyncio.sleep(self.retry_delay)
                            continue
                        return f"Error: Connection timeout after {self.max_retries} attempts"
                        
                    except Exception as e:
                        self.logger.error(f"Error with Ollama API: {str(e)}")
                        return f"Error: {str(e)}"
                
                # If we've exhausted all retries
                return "Error: Failed to generate text after multiple attempts"
    
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
                for attempt in range(self.max_retries):
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
                            if attempt < self.max_retries - 1:
                                self.logger.info(f"Retrying... (attempt {attempt+1}/{self.max_retries})")
                                await asyncio.sleep(self.retry_delay)
                                continue
                            # Append zeros as fallback
                            embeddings.append([0.0] * 768)  # Common embedding size
                            break
                            
                        response_data = response.json()
                        embeddings.append(response_data.get("embedding", []))
                        break  # Success, exit retry loop
                        
                    except (httpx.ConnectTimeout, httpx.ReadTimeout) as e:
                        self.logger.warning(f"Timeout error: {str(e)}")
                        if attempt < self.max_retries - 1:
                            self.logger.info(f"Retrying... (attempt {attempt+1}/{self.max_retries})")
                            await asyncio.sleep(self.retry_delay)
                            continue
                        # Append zeros as fallback
                        embeddings.append([0.0] * 768)  # Common embedding size
                        
                    except Exception as e:
                        self.logger.error(f"Error getting embeddings from Ollama: {str(e)}")
                        # Append zeros as fallback
                        embeddings.append([0.0] * 768)  # Common embedding size
                        break  # Exit retry loop for this text
        
        return embeddings
        
    async def generate_structured_output(
        self, 
        prompt: str, 
        schema: Union[Dict[str, Any], Type[BaseModel]], 
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Generate a structured JSON response using Ollama's API.
        
        Args:
            prompt: Text prompt to generate structured response for
            schema: Either a Pydantic model class or a JSON schema dictionary
            max_tokens: Maximum number of tokens to generate
            stream: Whether to enable streaming (forced to False for structured outputs)
            
        Returns:
            Structured data according to the provided schema
        """
        # Structured output doesn't support streaming
        if stream:
            self.logger.warning("Streaming not supported for structured outputs, falling back to non-streaming")
        
        # Extract schema from Pydantic model if provided
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            json_schema = schema.model_json_schema()
        else:
            json_schema = schema
            
        # Build instruction and system prompt
        schema_instruction = f"You must respond with a valid JSON object that conforms to this JSON schema: {json.dumps(json_schema)}"
        enhanced_system_prompt = f"{self.system_prompt or 'You are a helpful AI assistant.'}\n\n{schema_instruction}\n\nYour response should ONLY contain the JSON object and nothing else."
        
        # Build request payload for /api/chat endpoint
        # Ollama newer versions have more support for structured generation via chat
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": enhanced_system_prompt},
                {"role": "user", "content": prompt}
            ],
            "options": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "repeat_penalty": self.repeat_penalty
            },
            "format": "json"  # Request JSON format if supported by the model
        }
        
        if max_tokens or self.max_tokens:
            payload["options"]["num_predict"] = max_tokens if max_tokens else self.max_tokens
            
        # Try up to max_retries times
        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    # Try the chat endpoint first (newer Ollama versions)
                    response = await client.post(
                        f"{self.api_base_url}/api/chat",
                        json=payload,
                        timeout=60.0
                    )
                    
                    # If chat endpoint fails, fall back to generate endpoint
                    if response.status_code != 200:
                        self.logger.warning(f"Chat endpoint failed with {response.status_code}, falling back to generate endpoint")
                        # Modify payload for generate endpoint
                        generate_payload = {
                            "model": self.model_name,
                            "prompt": f"{enhanced_system_prompt}\n\nUser: {prompt}\n\nAssistant:",
                            "stream": False,
                            "options": payload["options"],
                            "format": "json"
                        }
                        
                        response = await client.post(
                            f"{self.api_base_url}/api/generate",
                            json=generate_payload,
                            timeout=60.0
                        )
                    
                    if response.status_code != 200:
                        error_detail = response.text
                        self.logger.error(f"Ollama API error: {response.status_code} - {error_detail}")
                        if attempt < self.max_retries - 1:
                            self.logger.info(f"Retrying... (attempt {attempt+1}/{self.max_retries})")
                            await asyncio.sleep(self.retry_delay)
                            continue
                        return {"error": f"API error: {response.status_code} - {error_detail}"}
                    
                    # Process the response based on endpoint
                    response_data = response.json()
                    
                    # Extract the JSON content from the response
                    content = ""
                    if "message" in response_data:
                        # Response from chat endpoint
                        content = response_data.get("message", {}).get("content", "")
                    else:
                        # Response from generate endpoint
                        content = response_data.get("response", "")
                    
                    # Parse the JSON content
                    try:
                        # Extract JSON if it's wrapped in backticks
                        if "```json" in content:
                            parts = content.split("```json")
                            if len(parts) > 1:
                                # Get the part after ```json
                                json_part = parts[1].split("```")[0].strip()
                                structured_data = json.loads(json_part)
                            else:
                                structured_data = json.loads(content)
                        elif "```" in content:
                            parts = content.split("```")
                            if len(parts) > 1:
                                # Get the part between ``` pairs
                                json_part = parts[1].strip()
                                structured_data = json.loads(json_part)
                            else:
                                structured_data = json.loads(content)
                        else:
                            # Try parsing the content directly
                            structured_data = json.loads(content)
                        
                        # Validate against schema if a Pydantic model was provided
                        if isinstance(schema, type) and issubclass(schema, BaseModel):
                            try:
                                # This will raise ValidationError if validation fails
                                validated_data = schema.model_validate(structured_data)
                                return validated_data.model_dump()
                            except Exception as e:
                                self.logger.error(f"Schema validation error: {str(e)}")
                                return {"error": f"Schema validation error: {str(e)}", "data": structured_data}
                        
                        return structured_data
                    
                    except json.JSONDecodeError as e:
                        self.logger.error(f"Failed to parse response as JSON: {e}\nContent: {content}")
                        
                        # Try a second attempt with a more aggressive JSON extraction
                        try:
                            # Look for anything that resembles a JSON object
                            import re
                            json_pattern = r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}'
                            matches = re.findall(json_pattern, content)
                            if matches:
                                structured_data = json.loads(matches[0])
                                self.logger.info("Successfully extracted JSON with regex")
                                return structured_data
                        except Exception:
                            pass
                            
                        if attempt < self.max_retries - 1:
                            self.logger.info(f"JSON parsing failed, retrying... (attempt {attempt+1}/{self.max_retries})")
                            await asyncio.sleep(self.retry_delay)
                            continue
                        
                        return {
                            "error": f"Failed to parse JSON: {str(e)}",
                            "raw_content": content
                        }
                
            except (httpx.ConnectTimeout, httpx.ReadTimeout) as e:
                self.logger.warning(f"Timeout error: {str(e)}")
                if attempt < self.max_retries - 1:
                    self.logger.info(f"Retrying... (attempt {attempt+1}/{self.max_retries})")
                    await asyncio.sleep(self.retry_delay)
                    continue
                return {"error": f"Connection timeout after {self.max_retries} attempts"}
                
            except Exception as e:
                self.logger.error(f"Error with Ollama API: {str(e)}")
                return {"error": str(e)}
        
        # If we've exhausted all retries
        return {"error": "Failed to generate structured output after multiple attempts"}
    
    def set_system_prompt(self, system_prompt: str):
        """Update the system prompt."""
        self.system_prompt = system_prompt