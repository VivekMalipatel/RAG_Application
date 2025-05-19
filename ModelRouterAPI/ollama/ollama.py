import logging
import json
import asyncio
from typing import List, Dict, Any, Optional, Union, AsyncGenerator, Type
import httpx
import time
from pydantic import BaseModel

from config import settings

class OllamaClient:
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
        self.model_name = hf_repo
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.stream = stream
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.stop = stop
        
        self.top_k = kwargs.get("top_k", 40)
        self.repeat_penalty = kwargs.get("repeat_penalty", 1.1)
        
        self.api_base_url = settings.OLLAMA_BASE_URL
        self.max_retries = kwargs.get("max_retries", 4) 
        self.retry_delay = kwargs.get("retry_delay", 2.0) 
        self.connection_timeout = kwargs.get("connection_timeout", 30.0)
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized Ollama client with model {hf_repo} at {self.api_base_url}")
        
        if not self.check_server_health():
            self.logger.warning(f"Ollama server at {self.api_base_url} is not responding. Some operations may fail.")
    
    def check_server_health(self) -> bool:
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
        for attempt in range(self.max_retries):
            try:
                with httpx.Client(timeout=self.connection_timeout) as client:
                    response = client.get(f"{self.api_base_url}/api/tags")
                    if response.status_code == 200:
                        models = response.json().get("models", [])
                        
                        model_names = [model.get("name", "") for model in models]
                        self.logger.info(f"Found {len(model_names)} Ollama models")
                        
                        if self.model_name in model_names:
                            self.logger.info(f"Found exact match for model: {self.model_name}")
                            return True
                        
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
        stream = stream if stream is not None else self.stream
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        temperature = temperature if temperature is not None else self.temperature
        top_p = top_p if top_p is not None else self.top_p
        stop = stop if stop is not None else self.stop
        
        model_name = self.model_name
        if ":" in model_name and not self.is_model_available():
            basic_model_name = model_name.split(":")[0]
            self.logger.info(f"Model {model_name} not found, trying with basic name {basic_model_name}")
            model_name = basic_model_name
        
        if isinstance(prompt, list):
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
        serializable_messages = []
        for message in messages:
            if hasattr(message, 'model_dump'):
                message = message.model_dump()

            if isinstance(message.get('content'), list):
                processed_message = {"role": message.get('role', 'user')}
                text_parts = []
                images = []
                
                for content_item in message['content']:
                    content_type = content_item.get('type')
                    
                    if content_type == 'text':
                        text_parts.append(content_item.get('text', ''))
                    elif content_type == 'image_url':
                        image_url = content_item.get('image_url', {}).get('url')
                        if image_url:
                            import base64
                            if image_url.startswith('data:image/'):
                                base64_part = image_url.split(',', 1)
                                if len(base64_part) > 1:
                                    images.append(base64_part[1])
                                else:
                                    self.logger.error(f"Invalid Base64 image format: {image_url[:30]}...")
                            elif image_url.startswith('http://') or image_url.startswith('https://'):
                                try:
                                    async with httpx.AsyncClient() as client:
                                        response = await client.get(image_url)
                                        if response.status_code == 200:
                                            base64_image = base64.b64encode(response.content).decode('utf-8')
                                            images.append(base64_image)
                                        else:
                                            self.logger.error(f"Failed to download image from {image_url}, status: {response.status_code}")
                                except Exception as e:
                                    self.logger.error(f"Error processing image {image_url}: {str(e)}")
                            else:
                                try:
                                    base64.b64decode(image_url)
                                    images.append(image_url)
                                except Exception as e:
                                    self.logger.error(f"Invalid Base64 string or URL: {image_url[:30]}... Error: {str(e)}")
                
                processed_message["content"] = " ".join(text_parts)
                
                if images:
                    processed_message["images"] = images
                
                serializable_messages.append(processed_message)
            else:
                serializable_messages.append(message)
        
        payload = {
            "model": model_name,
            "messages": serializable_messages,
            "stream": stream,
            "options": {
                "num_ctx": settings.OLLAMA_NUM_CTX,
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
            
        if stream:
            async def generate_stream():
                client = httpx.AsyncClient(timeout=7200.0)
                try:
                    for attempt in range(self.max_retries):
                        try:
                            async with client.stream(
                                "POST", 
                                f"{self.api_base_url}/api/chat",
                                json=payload, 
                                timeout=7200.0
                            ) as response:
                                if response.status_code != 200:
                                    if attempt < self.max_retries - 1:
                                        await asyncio.sleep(self.retry_delay)
                                        continue
                                    yield ""
                                    return
                                    
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
        
        else:
            async with httpx.AsyncClient(timeout=7200.0) as client:
                for attempt in range(self.max_retries):
                    try:
                        response = await client.post(
                            f"{self.api_base_url}/api/chat",
                            json=payload,
                            timeout=7200.0
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
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "num_ctx": settings.OLLAMA_NUM_CTX,
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
        
        if stream:
            async def generate_stream():
                client = httpx.AsyncClient(timeout=7200.0)
                try:
                    for attempt in range(self.max_retries):
                        try:
                            async with client.stream(
                                "POST", 
                                f"{self.api_base_url}/api/generate",
                                json=payload, 
                                timeout=7200.0
                            ) as response:
                                if response.status_code != 200:
                                    if attempt < self.max_retries - 1:
                                        await asyncio.sleep(self.retry_delay)
                                        continue
                                    yield ""
                                    return
                                    
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
        
        else:
            async with httpx.AsyncClient(timeout=7200.0) as client:
                for attempt in range(self.max_retries):
                    try:
                        response = await client.post(
                            f"{self.api_base_url}/api/generate",
                            json=payload,
                            timeout=7200.0
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
                            embeddings.append([0.0] * 768)
                            break
                            
                        response_data = response.json()
                        embeddings.append(response_data.get("embedding", []))
                        break
                        
                    except (httpx.ConnectTimeout, httpx.ReadTimeout) as e:
                        self.logger.warning(f"Timeout error: {str(e)}")
                        if attempt < self.max_retries - 1:
                            self.logger.info(f"Retrying... (attempt {attempt+1}/{self.max_retries})")
                            await asyncio.sleep(self.retry_delay)
                            continue
                        embeddings.append([0.0] * 768)
                        
                    except Exception as e:
                        self.logger.error(f"Error getting embeddings from Ollama: {str(e)}")
                        embeddings.append([0.0] * 768)
                        break
        
        return embeddings
        
    async def generate_structured_output(
        self, 
        prompt: str, 
        schema: Union[Dict[str, Any], Type[BaseModel]], 
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        if stream:
            self.logger.warning("Streaming is not supported for structured JSON output with Ollama, using non-streaming.")
        
        is_pydantic_schema = isinstance(schema, type) and issubclass(schema, BaseModel)
        if is_pydantic_schema:
            json_schema_def = schema.model_json_schema()
        else:
            json_schema_def = schema
            
        options = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repeat_penalty": self.repeat_penalty
        }
        if max_tokens or self.max_tokens:
            options["num_predict"] = max_tokens if max_tokens else self.max_tokens
            
        generate_prompt = f"{self.system_prompt or 'You are a helpful AI assistant.'}\n\nUser: {prompt}\n\nAssistant (Respond using JSON conforming to the provided schema):"
        
        generate_payload = {
            "model": self.model_name,
            "prompt": generate_prompt,
            "options": options,
            "format": json_schema_def,
            "stream": False
        }

        last_error = None
        last_response = None

        for attempt in range(self.max_retries):
            response = None
            response_data = None
            try:
                async with httpx.AsyncClient(timeout=7200.0) as client:
                    self.logger.debug(f"Attempting structured output via /api/generate (attempt {attempt+1}/{self.max_retries})")
                    response = await client.post(
                        f"{self.api_base_url}/api/generate",
                        json=generate_payload,
                        timeout=7200.0
                    )
                    last_response = response

                    if response.status_code == 200:
                        response_data = response.json()
                        self.logger.info("Received successful response from /api/generate.")
                        break
                    else:
                        last_error = f"API returned status {response.status_code}"
                        self.logger.warning(f"/api/generate failed with status {response.status_code}. Response: {response.text}")

            except (httpx.ConnectTimeout, httpx.ReadTimeout) as e:
                last_error = f"Timeout contacting API: {e}"
                self.logger.warning(f"{last_error} on attempt {attempt+1}")
            except Exception as e:
                last_error = f"Unexpected error contacting API: {e}"
                self.logger.error(f"{last_error} on attempt {attempt+1}", exc_info=True)

            if attempt < self.max_retries - 1:
                self.logger.info(f"Retrying API call for structured generation... (attempt {attempt+1}/{self.max_retries})")
                await asyncio.sleep(self.retry_delay)
            else:
                error_msg = f"Failed to get successful response after {self.max_retries} attempts. Last error: {last_error}"
                if last_response:
                    try:
                        error_msg += f" Last status: {last_response.status_code} - {last_response.text}"
                    except Exception:
                         error_msg += f" Last status: {last_response.status_code}"
                self.logger.error(error_msg)
                return {"error": error_msg}

        if response_data:
            content = response_data.get("response", "")
            self.logger.debug(f"Raw content received: {content}")
            content = content.strip()

            structured_data = None
            parse_error = None
            try:
                structured_data = json.loads(content)
                self.logger.debug("Successfully parsed JSON directly.")
            except json.JSONDecodeError as e1:
                self.logger.warning(f"Direct JSON parsing failed: {e1}. Attempting extraction.")
                try:
                    start_index = content.find('{')
                    end_index = content.rfind('}')
                    if start_index != -1 and end_index != -1 and start_index < end_index:
                        json_substring = content[start_index : end_index + 1]
                        structured_data = json.loads(json_substring)
                        self.logger.info("Successfully parsed JSON after extraction.")
                    else:
                        parse_error = e1
                        self.logger.error("Could not find valid JSON object boundaries '{...}'.")
                except json.JSONDecodeError as e2:
                    parse_error = e2
                    self.logger.error(f"JSON extraction parsing failed: {e2}")

            if structured_data is None:
                self.logger.error(f"Failed to parse response as JSON. Error: {parse_error}")
                return {
                    "error": f"Failed to parse JSON: {parse_error}",
                    "raw_content": content
                }
            else:
                if is_pydantic_schema:
                    try:
                        validated_data = schema.model_validate(structured_data)
                        self.logger.info("JSON response successfully parsed and validated.")
                        return validated_data.model_dump()
                    except Exception as validation_error:
                        self.logger.error(f"Schema validation failed: {validation_error}")
                        return {
                            "error": f"Schema validation failed: {validation_error}",
                            "parsed_data": structured_data
                        }
                else:
                    self.logger.info("JSON response successfully parsed (no Pydantic validation performed).")
                    return structured_data
        else:
             self.logger.error("API call succeeded but response_data is missing.")
             return {"error": "Internal error: Missing response data after successful API call."}


    def set_system_prompt(self, system_prompt: str):
        self.system_prompt = system_prompt

    async def list_available_models(self) -> List[str]:
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