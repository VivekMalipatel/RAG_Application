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
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: Optional[Union[str, List[str]]] = None,
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
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.stop = stop
        
        # Extract additional parameters
        self.top_k = kwargs.get("top_k", 40)
        self.repeat_penalty = kwargs.get("repeat_penalty", 1.1)
        
        # Connection settings - increased for more reliable connections
        self.api_base_url = settings.OLLAMA_BASE_URL
        self.max_retries = kwargs.get("max_retries", 4) 
        self.retry_delay = kwargs.get("retry_delay", 2.0) 
        self.connection_timeout = kwargs.get("connection_timeout", 30.0)
        
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
                        
                        # Get all model names
                        model_names = [model.get("name", "") for model in models]
                        self.logger.info(f"Found {len(model_names)} Ollama models")
                        
                        # Check for exact match
                        if self.model_name in model_names:
                            self.logger.info(f"Found exact match for model: {self.model_name}")
                            return True
                        
                        # Get just the base model names without tags
                        base_model_names = [name.split(':')[0] for name in model_names]
                        if self.model_name.split(':')[0] in base_model_names:
                            self.logger.info(f"Found base model match for: {self.model_name}")
                            return True
                            
                        self.logger.warning(f"Model {self.model_name} not found in available models")
                        return False
                    
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
        prompt: Union[str, List[Dict[str, Any]]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None,
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
        temperature = temperature if temperature is not None else self.temperature
        top_p = top_p if top_p is not None else self.top_p
        stop = stop if stop is not None else self.stop
        
        # Use simplified model name if requested
        model_name = self.model_name
        # Strip off any tag/version to try a more basic model name
        if ":" in model_name and not self.is_model_available():
            basic_model_name = model_name.split(":")[0]
            self.logger.info(f"Model {model_name} not found, trying with basic name {basic_model_name}")
            model_name = basic_model_name
        
        # Check if prompt is in message format
        if isinstance(prompt, list):
            # Handle messages format by using chat endpoint
            return await self._generate_from_messages(
                messages=prompt,
                model_name=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                stream=stream
            )
        else:
            # Regular prompt handling with generate endpoint
            return await self._generate_from_prompt(
                prompt=prompt,
                model_name=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                stream=stream
            )
    
    async def _generate_from_messages(
        self, 
        messages: List[Dict[str, Any]],
        model_name: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None,
        stream: bool = False
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Generate text from messages format using Ollama's chat API"""
        
        # Convert any Pydantic model objects to dictionaries
        serializable_messages = []
        for message in messages:
            if hasattr(message, 'model_dump'):
                serializable_messages.append(message.model_dump())
            else:
                serializable_messages.append(message)
        
        payload = {
            "model": model_name,
            "messages": serializable_messages,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": self.top_k,
                "repeat_penalty": self.repeat_penalty
            }
        }
        
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens
            
        if stop:
            payload["options"]["stop"] = stop if isinstance(stop, list) else [stop]
            
        # Handle streaming
        if stream:
            async def generate_stream():
                client = httpx.AsyncClient(timeout=60.0)
                try:
                    for attempt in range(self.max_retries):
                        try:
                            async with client.stream(
                                "POST", 
                                f"{self.api_base_url}/api/chat",
                                json=payload, 
                                timeout=60.0
                            ) as response:
                                if response.status_code != 200:
                                    if attempt < self.max_retries - 1:
                                        await asyncio.sleep(self.retry_delay)
                                        continue
                                    yield ""
                                    return
                                    
                                # Stream the response
                                async for line in response.aiter_lines():
                                    if line:
                                        try:
                                            chunk_data = json.loads(line)
                                            if "message" in chunk_data and "content" in chunk_data["message"]:
                                                yield chunk_data["message"]["content"]
                                        except json.JSONDecodeError:
                                            pass
                                
                                break
                                
                        except (httpx.ConnectTimeout, httpx.ReadTimeout):
                            if attempt < self.max_retries - 1:
                                await asyncio.sleep(self.retry_delay)
                                continue
                            yield ""
                            return
                        except Exception:
                            yield ""
                            return
                finally:
                    await client.aclose()
            
            return generate_stream()
        
        # Non-streaming
        else:
            async with httpx.AsyncClient(timeout=60.0) as client:
                for attempt in range(self.max_retries):
                    try:
                        response = await client.post(
                            f"{self.api_base_url}/api/chat",
                            json=payload,
                            timeout=60.0
                        )
                        
                        if response.status_code != 200:
                            if attempt < self.max_retries - 1:
                                await asyncio.sleep(self.retry_delay)
                                continue
                            return ""
                            
                        response_data = response.json()
                        if "message" in response_data and "content" in response_data["message"]:
                            return response_data["message"]["content"]
                        return ""
                        
                    except (httpx.ConnectTimeout, httpx.ReadTimeout) as e:
                        self.logger.error(f"Connection timeout to Ollama API: {str(e)}")
                        if attempt < self.max_retries - 1:
                            await asyncio.sleep(self.retry_delay)
                            continue
                        self.logger.error(f"Failed to connect to Ollama API after {self.max_retries} attempts")
                        return ""
                        
                    except Exception as e:
                        self.logger.error(f"Error connecting to Ollama API: {str(e)}")
                        return ""
                
                return ""
    
    async def _generate_from_prompt(
        self, 
        prompt: str,
        model_name: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None,
        stream: bool = False
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Generate text from a single prompt using Ollama's generate API"""
        # Build request payload
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": self.top_k,
                "repeat_penalty": self.repeat_penalty
            }
        }
        
        if self.system_prompt:
            payload["system"] = self.system_prompt
            
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens
            
        if stop:
            payload["options"]["stop"] = stop if isinstance(stop, list) else [stop]
        
        # Handle streaming
        if stream:
            async def generate_stream():
                client = httpx.AsyncClient(timeout=60.0)
                try:
                    for attempt in range(self.max_retries):
                        try:
                            async with client.stream(
                                "POST", 
                                f"{self.api_base_url}/api/generate",
                                json=payload, 
                                timeout=60.0
                            ) as response:
                                if response.status_code != 200:
                                    if attempt < self.max_retries - 1:
                                        await asyncio.sleep(self.retry_delay)
                                        continue
                                    yield ""
                                    return
                                    
                                # Stream the response
                                async for line in response.aiter_lines():
                                    if line:
                                        try:
                                            chunk_data = json.loads(line)
                                            if "response" in chunk_data:
                                                yield chunk_data["response"]
                                        except json.JSONDecodeError:
                                            pass
                                
                                break
                                
                        except (httpx.ConnectTimeout, httpx.ReadTimeout):
                            if attempt < self.max_retries - 1:
                                await asyncio.sleep(self.retry_delay)
                                continue
                            yield ""
                            return
                        except Exception:
                            yield ""
                            return
                finally:
                    await client.aclose()
            
            return generate_stream()
        
        # Non-streaming
        else:
            async with httpx.AsyncClient(timeout=60.0) as client:
                for attempt in range(self.max_retries):
                    try:
                        response = await client.post(
                            f"{self.api_base_url}/api/generate",
                            json=payload,
                            timeout=60.0
                        )
                        
                        if response.status_code != 200:
                            if attempt < self.max_retries - 1:
                                await asyncio.sleep(self.retry_delay)
                                continue
                            return ""
                            
                        response_data = response.json()
                        return response_data.get("response", "")
                        
                    except (httpx.ConnectTimeout, httpx.ReadTimeout):
                        if attempt < self.max_retries - 1:
                            await asyncio.sleep(self.retry_delay)
                            continue
                        return ""
                        
                    except Exception:
                        return ""
                
                return ""
    
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
        stream: bool = False # Note: stream is effectively ignored for structured output
    ) -> Dict[str, Any]:
        """
        Generate a structured JSON response using Ollama's /api/generate endpoint,
        passing the schema directly in the format parameter. Parses the response and
        optionally validates against a Pydantic schema.
        
        Args:
            prompt: Text prompt to generate structured response for
            schema: Either a Pydantic model class or a JSON schema dictionary.
            max_tokens: Maximum number of tokens to generate
            stream: Whether to enable streaming (forced to False for structured outputs)
            
        Returns:
            A dictionary containing the parsed (and potentially validated) JSON data, 
            or an error dictionary on failure.
        """
        if stream:
            self.logger.warning("Streaming is not supported for structured JSON output with Ollama, using non-streaming.")
        
        is_pydantic_schema = isinstance(schema, type) and issubclass(schema, BaseModel)
        # Extract schema definition
        if is_pydantic_schema:
            json_schema_def = schema.model_json_schema()
        else:
            json_schema_def = schema # Assume it's already a dict
            
        # Common options
        options = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repeat_penalty": self.repeat_penalty
        }
        if max_tokens or self.max_tokens:
            options["num_predict"] = max_tokens if max_tokens else self.max_tokens
            
        # Construct prompt suitable for generate endpoint with schema in format
        generate_prompt = f"{self.system_prompt or 'You are a helpful AI assistant.'}\n\nUser: {prompt}\n\nAssistant (Respond using JSON conforming to the provided schema):"
        
        # Construct payload for /api/generate
        generate_payload = {
            "model": self.model_name,
            "prompt": generate_prompt,
            "options": options,
            "format": json_schema_def, # Pass the schema dictionary directly
            "stream": False
        }

        last_error = None
        last_response = None

        # Try up to max_retries times for the API call
        for attempt in range(self.max_retries):
            response = None
            response_data = None
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    self.logger.debug(f"Attempting structured output via /api/generate (attempt {attempt+1}/{self.max_retries})")
                    response = await client.post(
                        f"{self.api_base_url}/api/generate",
                        json=generate_payload,
                        timeout=60.0
                    )
                    last_response = response # Store last response for error reporting

                    if response.status_code == 200:
                        response_data = response.json()
                        self.logger.info("Received successful response from /api/generate.")
                        break # Exit retry loop on successful API call
                    else:
                        last_error = f"API returned status {response.status_code}"
                        self.logger.warning(f"/api/generate failed with status {response.status_code}. Response: {response.text}")
                        # Continue to retry logic

            except (httpx.ConnectTimeout, httpx.ReadTimeout) as e:
                last_error = f"Timeout contacting API: {e}"
                self.logger.warning(f"{last_error} on attempt {attempt+1}")
                # Continue to retry logic
            except Exception as e:
                last_error = f"Unexpected error contacting API: {e}"
                self.logger.error(f"{last_error} on attempt {attempt+1}", exc_info=True)
                # Continue to retry logic

            # If this wasn't the last attempt, wait before retrying
            if attempt < self.max_retries - 1:
                self.logger.info(f"Retrying API call for structured generation... (attempt {attempt+1}/{self.max_retries})")
                await asyncio.sleep(self.retry_delay)
            else:
                # Last attempt failed API call
                error_msg = f"Failed to get successful response after {self.max_retries} attempts. Last error: {last_error}"
                if last_response:
                    try:
                        error_msg += f" Last status: {last_response.status_code} - {last_response.text}"
                    except Exception:
                         error_msg += f" Last status: {last_response.status_code}"
                self.logger.error(error_msg)
                return {"error": error_msg}

        # --- Process successful response ---
        if response_data:
            content = response_data.get("response", "")
            self.logger.debug(f"Raw content received: {content}")
            content = content.strip()

            # Attempt to parse the JSON content
            structured_data = None
            parse_error = None
            try:
                # First, try direct parsing
                structured_data = json.loads(content)
                self.logger.debug("Successfully parsed JSON directly.")
            except json.JSONDecodeError as e1:
                self.logger.warning(f"Direct JSON parsing failed: {e1}. Attempting extraction.")
                # If direct parsing fails, try extracting from potential wrapping characters
                try:
                    start_index = content.find('{')
                    end_index = content.rfind('}')
                    if start_index != -1 and end_index != -1 and start_index < end_index:
                        json_substring = content[start_index : end_index + 1]
                        structured_data = json.loads(json_substring)
                        self.logger.info("Successfully parsed JSON after extraction.")
                    else:
                        parse_error = e1 # Keep original error if extraction indices are bad
                        self.logger.error("Could not find valid JSON object boundaries '{...}'.")
                except json.JSONDecodeError as e2:
                    parse_error = e2 # Keep the error from the second parse attempt
                    self.logger.error(f"JSON extraction parsing failed: {e2}")

            # If parsing failed after all attempts
            if structured_data is None:
                self.logger.error(f"Failed to parse response as JSON. Error: {parse_error}")
                return {
                    "error": f"Failed to parse JSON: {parse_error}",
                    "raw_content": content
                }
            else:
                # --- Validation (Optional but recommended) ---
                if is_pydantic_schema:
                    try:
                        # Validate the parsed data against the Pydantic model
                        validated_data = schema.model_validate(structured_data)
                        self.logger.info("JSON response successfully parsed and validated.")
                        return validated_data.model_dump() # Return validated and serialized data
                    except Exception as validation_error:
                        # Log validation error but return the parsed data anyway, flagging the error
                        self.logger.error(f"Schema validation failed: {validation_error}")
                        return {
                            "error": f"Schema validation failed: {validation_error}",
                            "parsed_data": structured_data # Return the data that was parsed
                        }
                else:
                    # If no Pydantic schema provided, return the parsed JSON directly
                    self.logger.info("JSON response successfully parsed (no Pydantic validation performed).")
                    return structured_data
        else:
             # Should be unreachable if API call succeeded but response_data is None
             self.logger.error("API call succeeded but response_data is missing.")
             return {"error": "Internal error: Missing response data after successful API call."}


    def set_system_prompt(self, system_prompt: str):
        """Update the system prompt."""
        self.system_prompt = system_prompt

    async def list_available_models(self) -> List[str]:
        """List all available models in Ollama server."""
        try:
            async with httpx.AsyncClient(timeout=self.connection_timeout) as client:
                response = await client.get(f"{self.api_base_url}/api/tags")
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    model_names = [model.get("name", "") for model in models]
                    self.logger.info(f"Found {len(model_names)} available Ollama models")
                    return model_names
                self.logger.warning(f"Failed to get models, status: {response.status_code}")
                return []
        except Exception as e:
            self.logger.error(f"Error listing available models: {str(e)}")
            return []