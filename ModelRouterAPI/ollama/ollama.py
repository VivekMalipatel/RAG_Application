import logging
import json
import asyncio
import base64
from typing import List, Dict, Any, Optional, Union, AsyncGenerator, Type
import httpx
import time
from pydantic import BaseModel
from config import settings
from schemas.chat import ChatMessage, ChatCompletionMessageToolCall, ChatCompletionMessageToolCallFunction

class OllamaClient:
    def __init__(
        self,
        hf_repo: str,
        system_prompt: Optional[str] = None,
        temperature: float = None,
        top_p: float = None,
        max_tokens: Optional[int] = None,
        max_completion_tokens: Optional[int] = None,
        stream: bool = None,
        frequency_penalty: float = None,
        presence_penalty: float = None,
        stop: Optional[Union[str, List[str]]] = None,
        logit_bias: Optional[Dict[str, float]] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        n: Optional[int] = None,
        seed: Optional[int] = None,
        user: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        parallel_tool_calls: Optional[bool] = None,
        response_format: Optional[Dict[str, Any]] = None,
        service_tier: Optional[str] = None,
        store: Optional[bool] = None,
        metadata: Optional[Dict[str, str]] = None,
        reasoning_effort: Optional[str] = None,
        modalities: Optional[List[str]] = None,
        audio: Optional[Dict[str, Any]] = None,
        prediction: Optional[Dict[str, Any]] = None,
        web_search_options: Optional[Dict[str, Any]] = None,
        stream_options: Optional[Dict[str, Any]] = None,
        num_ctx: Optional[int] = None,
        repeat_last_n: Optional[int] = None,
        repeat_penalty: Optional[float] = None,
        top_k: Optional[int] = None,
        min_p: Optional[float] = None,
        keep_alive: Optional[str] = None,
        think: Optional[bool] = None,
        **kwargs
    ):
        self.model_name = hf_repo
        self.system_prompt = system_prompt
        self.temperature = temperature if temperature is not None else settings.OLLAMA_DEFAULT_TEMPERATURE
        self.top_p = top_p if top_p is not None else settings.OLLAMA_DEFAULT_TOP_P
        self.max_tokens = max_tokens
        self.max_completion_tokens = max_completion_tokens
        self.stream = stream if stream is not None else settings.OLLAMA_DEFAULT_STREAM
        self.frequency_penalty = frequency_penalty if frequency_penalty is not None else settings.OLLAMA_DEFAULT_FREQUENCY_PENALTY
        self.presence_penalty = presence_penalty if presence_penalty is not None else settings.OLLAMA_DEFAULT_PRESENCE_PENALTY
        self.stop = stop
        self.logit_bias = logit_bias
        self.logprobs = logprobs
        self.top_logprobs = top_logprobs
        self.n = n if n is not None else settings.OLLAMA_DEFAULT_N
        self.seed = seed
        self.user = user
        self.tools = tools
        self.tool_choice = tool_choice
        self.parallel_tool_calls = parallel_tool_calls if parallel_tool_calls is not None else settings.OLLAMA_DEFAULT_PARALLEL_TOOL_CALLS
        self.response_format = response_format
        self.service_tier = service_tier
        self.store = store
        self.metadata = metadata
        self.reasoning_effort = reasoning_effort
        self.modalities = modalities
        self.audio = audio
        self.prediction = prediction
        self.web_search_options = web_search_options
        self.stream_options = stream_options
        
        self.num_ctx = num_ctx or kwargs.get("num_ctx")
        self.repeat_last_n = repeat_last_n or kwargs.get("repeat_last_n")
        self.repeat_penalty = repeat_penalty or kwargs.get("repeat_penalty")
        self.top_k = top_k or kwargs.get("top_k")
        self.min_p = min_p or kwargs.get("min_p")
        self.keep_alive = keep_alive or kwargs.get("keep_alive")
        self.think = think or kwargs.get("think")

        self.api_base_url = settings.OLLAMA_BASE_URL
        self.max_retries = kwargs.get("max_retries", 1)
        self.retry_delay = kwargs.get("retry_delay", 5)
        self.connection_timeout = kwargs.get("connection_timeout", 10)

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized Ollama client with model {hf_repo} at {self.api_base_url}")
        
        if self.frequency_penalty != 0.0:
            self.logger.warning("frequency_penalty mapped to repeat_penalty for Ollama compatibility")
            if repeat_penalty is None:
                self.repeat_penalty = 1.0 + self.frequency_penalty
        
        if self.n > 1:
            self.logger.warning("Multiple completions (n > 1) not supported by Ollama, will return single completion")
        
        unsupported_params = [
            "logit_bias", "logprobs", "top_logprobs", "presence_penalty", 
            "service_tier", "store", "metadata", "reasoning_effort", 
            "audio", "prediction", "web_search_options", "user"
        ]
        for param in unsupported_params:
            if getattr(self, param) is not None:
                self.logger.warning(f"{param} is not supported by Ollama and will be ignored")
        
        if not self.check_server_health():
            error_msg = f"Ollama server at {self.api_base_url} is not responding"
            self.logger.error(error_msg)
            raise ConnectionError(error_msg)
    
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
        max_completion_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None,
        stream: Optional[bool] = None,
        logit_bias: Optional[Dict[str, float]] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        n: Optional[int] = None,
        seed: Optional[int] = None,
        user: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        parallel_tool_calls: Optional[bool] = None,
        response_format: Optional[Dict[str, Any]] = None,
        service_tier: Optional[str] = None,
        store: Optional[bool] = None,
        metadata: Optional[Dict[str, str]] = None,
        reasoning_effort: Optional[str] = None,
        modalities: Optional[List[str]] = None,
        audio: Optional[Dict[str, Any]] = None,
        prediction: Optional[Dict[str, Any]] = None,
        web_search_options: Optional[Dict[str, Any]] = None,
        stream_options: Optional[Dict[str, Any]] = None,
        num_ctx: Optional[int] = None,
        repeat_last_n: Optional[int] = None,
        repeat_penalty: Optional[float] = None,
        top_k: Optional[int] = None,
        min_p: Optional[float] = None,
        keep_alive: Optional[str] = None,
        think: Optional[bool] = None
    ) -> Union[str, List[str], AsyncGenerator[str, None]]:
        stream = stream if stream is not None else self.stream
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        max_completion_tokens = max_completion_tokens if max_completion_tokens is not None else self.max_completion_tokens
        temperature = temperature if temperature is not None else self.temperature
        top_p = top_p if top_p is not None else self.top_p
        stop = stop if stop is not None else self.stop
        seed = seed if seed is not None else self.seed
        tools = tools if tools is not None else self.tools
        tool_choice = tool_choice if tool_choice is not None else self.tool_choice
        response_format = response_format if response_format is not None else self.response_format
        num_ctx = num_ctx if num_ctx is not None else self.num_ctx
        repeat_last_n = repeat_last_n if repeat_last_n is not None else self.repeat_last_n
        repeat_penalty = repeat_penalty if repeat_penalty is not None else self.repeat_penalty
        top_k = top_k if top_k is not None else self.top_k
        min_p = min_p if min_p is not None else self.min_p
        keep_alive = keep_alive if keep_alive is not None else self.keep_alive
        think = think if think is not None else self.think
        
        num_predict = max_completion_tokens or max_tokens
        
        if n and n > 1:
            self.logger.warning("Multiple completions (n > 1) not supported by Ollama, returning single completion")
        
        unsupported_in_call = [logit_bias, logprobs, top_logprobs, user, service_tier, store, metadata, reasoning_effort, audio, prediction, web_search_options]
        if any(param is not None for param in unsupported_in_call):
            self.logger.warning("Some parameters are not supported by Ollama and will be ignored")
        
        model_name = self.model_name
        if ":" in model_name and not self.is_model_available():
            basic_model_name = model_name.split(":")[0]
            self.logger.info(f"Model {model_name} not found, trying with basic name {basic_model_name}")
            model_name = basic_model_name
        
        if isinstance(prompt, list):
            return await self._generate_from_messages(
                messages=prompt,
                model_name=model_name,
                num_predict=num_predict,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                stream=stream,
                seed=seed,
                tools=tools,
                tool_choice=tool_choice,
                response_format=response_format,
                num_ctx=num_ctx,
                repeat_last_n=repeat_last_n,
                repeat_penalty=repeat_penalty,
                top_k=top_k,
                min_p=min_p,
                keep_alive=keep_alive,
                think=think
            )
        else:
            return await self._generate_from_prompt(
                prompt=prompt,
                model_name=model_name,
                num_predict=num_predict,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                stream=stream,
                seed=seed,
                tools=tools,
                tool_choice=tool_choice,
                response_format=response_format,
                num_ctx=num_ctx,
                repeat_last_n=repeat_last_n,
                repeat_penalty=repeat_penalty,
                top_k=top_k,
                min_p=min_p,
                keep_alive=keep_alive,
                think=think
            )
    
    async def _generate_from_messages(
        self, 
        messages: List[Dict[str, Any]],
        model_name: str,
        num_predict: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None,
        stream: bool = False,
        seed: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        num_ctx: Optional[int] = None,
        repeat_last_n: Optional[int] = None,
        repeat_penalty: Optional[float] = None,
        top_k: Optional[int] = None,
        min_p: Optional[float] = None,
        keep_alive: Optional[str] = None,
        think: Optional[bool] = None
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
                
                if message.get('thinking'):
                    processed_message["thinking"] = message['thinking']
                
                if message.get('tool_calls'):
                    processed_message["tool_calls"] = message['tool_calls']
                
                serializable_messages.append(processed_message)
            else:
                filtered_message = {"role": message.get('role', 'user')}
                
                if message.get('content'):
                    filtered_message["content"] = message['content']
                
                if message.get('thinking'):
                    filtered_message["thinking"] = message['thinking']
                
                if message.get('tool_calls'):
                    filtered_message["tool_calls"] = message['tool_calls']
                
                serializable_messages.append(filtered_message)
        
        options = {
            "num_ctx": num_ctx or self.num_ctx,
            "temperature": temperature or self.temperature,
            "top_p": top_p or self.top_p,
            "top_k": top_k or self.top_k,
            "repeat_penalty": repeat_penalty or self.repeat_penalty
        }
        
        if num_predict:
            options["num_predict"] = num_predict
        if repeat_last_n is not None:
            options["repeat_last_n"] = repeat_last_n
        if min_p is not None:
            options["min_p"] = min_p
        if seed is not None:
            options["seed"] = seed
        if stop:
            options["stop"] = stop if isinstance(stop, list) else [stop]
        
        payload = {
            "model": model_name,
            "messages": serializable_messages,
            "stream": stream,
            "options": options
        }
        
        # if response_format:
        #     if isinstance(response_format, dict) and response_format.get("type") == "json_object":
        #         self.logger.info("Using structured JSON output format")
        #         payload["format"] = "json"
        #     elif isinstance(response_format, dict) and "json_schema" in response_format:
        #         schema = response_format["json_schema"].get("schema", {})
        #         payload["format"] = schema
        
        if tools and len(tools) > 0:
            self.logger.warning("Tools support in Ollama is experimental and may not work as expected")
            payload["tools"] = tools
            if tool_choice:
                payload["tool_choice"] = tool_choice
        
        if keep_alive:
            payload["keep_alive"] = keep_alive
        
        if think is not None:
            options["think"] = think
            
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
                                    raise httpx.HTTPStatusError(
                                        f"Ollama API returned status {response.status_code}",
                                        request=response.request,
                                        response=response
                                    )
                                    
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
                            raise httpx.TimeoutException(f"Timeout connecting to Ollama API after {self.max_retries} attempts")
                        except Exception as e:
                            if attempt < self.max_retries - 1:
                                await asyncio.sleep(self.retry_delay)
                                continue
                            raise RuntimeError(f"Error connecting to Ollama API: {str(e)}")
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
                            print(response.status_code, response.text)
                            if attempt < self.max_retries - 1:
                                await asyncio.sleep(self.retry_delay)
                                continue
                            raise httpx.HTTPStatusError(
                                f"Ollama API returned status {response.status_code}: {response.text}",
                                request=response.request,
                                response=response
                            )
                            
                        response_data = response.json()
                        if "message" in response_data:
                            message_data = response_data["message"]
                            
                            # Create ChatMessage object
                            chat_message = ChatMessage(
                                role="assistant",
                                content=message_data.get("content", ""),
                                tool_calls=None
                            )
                            
                            # Parse tool calls if present
                            if "tool_calls" in message_data and message_data["tool_calls"]:
                                tool_calls = []
                                for tool_call in message_data["tool_calls"]:
                                    if "function" in tool_call:
                                        function_data = tool_call["function"]
                                        tool_call_obj = ChatCompletionMessageToolCall(
                                            id=tool_call.get("id", f"call_{int(time.time() * 1000)}"),
                                            type="function",
                                            function=ChatCompletionMessageToolCallFunction(
                                                name=function_data.get("name", ""),
                                                arguments=json.dumps(function_data.get("arguments", {})) if isinstance(function_data.get("arguments"), dict) else str(function_data.get("arguments", "{}"))
                                            )
                                        )
                                        tool_calls.append(tool_call_obj)
                                chat_message.tool_calls = tool_calls
                            
                            return chat_message
                        raise ValueError("Invalid response format from Ollama API: missing message")
                        
                    except (httpx.ConnectTimeout, httpx.ReadTimeout) as e:
                        self.logger.error(f"Connection timeout to Ollama API: {str(e)}")
                        if attempt < self.max_retries - 1:
                            await asyncio.sleep(self.retry_delay)
                            continue
                        self.logger.error(f"Failed to connect to Ollama API after {self.max_retries} attempts")
                        raise httpx.TimeoutException(f"Failed to connect to Ollama API after {self.max_retries} attempts: {str(e)}")
                        
                    except Exception as e:
                        self.logger.error(f"Error connecting to Ollama API: {str(e)}")
                        if attempt < self.max_retries - 1:
                            await asyncio.sleep(self.retry_delay)
                            continue
                        raise RuntimeError(f"Error connecting to Ollama API: {str(e)}")
                
                raise RuntimeError("Failed to get response from Ollama API after all retry attempts")
    
    async def _generate_from_prompt(
        self, 
        prompt: str,
        model_name: str,
        num_predict: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None,
        stream: bool = False,
        seed: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        num_ctx: Optional[int] = None,
        repeat_last_n: Optional[int] = None,
        repeat_penalty: Optional[float] = None,
        top_k: Optional[int] = None,
        min_p: Optional[float] = None,
        keep_alive: Optional[str] = None,
        think: Optional[bool] = None
    ) -> Union[str, AsyncGenerator[str, None]]:
        options = {
            "num_ctx": num_ctx or settings.OLLAMA_NUM_CTX,
            "temperature": temperature or self.temperature,
            "top_p": top_p or self.top_p,
            "top_k": top_k or self.top_k,
            "repeat_penalty": repeat_penalty or self.repeat_penalty
        }
        
        if num_predict:
            options["num_predict"] = num_predict
        if repeat_last_n is not None:
            options["repeat_last_n"] = repeat_last_n
        if min_p is not None:
            options["min_p"] = min_p
        if seed is not None:
            options["seed"] = seed
        if stop:
            options["stop"] = stop if isinstance(stop, list) else [stop]
        
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": stream,
            "options": options
        }
        
        if self.system_prompt:
            payload["system"] = self.system_prompt
        
        # if response_format:
        #     if isinstance(response_format, dict) and response_format.get("type") == "json_object":
        #         payload["format"] = "json"
        #     elif isinstance(response_format, dict) and "json_schema" in response_format:
        #         schema = response_format["json_schema"].get("schema", {})
        #         payload["format"] = schema
        
        if tools and len(tools) > 0:
            self.logger.warning("Tools support in Ollama is experimental and may not work as expected")
            payload["tools"] = tools
            if tool_choice:
                payload["tool_choice"] = tool_choice
        
        if keep_alive:
            payload["keep_alive"] = keep_alive
        
        if think is not None:
            payload["think"] = think
        
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
                                    raise httpx.HTTPStatusError(
                                        f"Ollama API returned status {response.status_code}",
                                        request=response.request,
                                        response=response
                                    )
                                    
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
                            raise httpx.TimeoutException(f"Timeout connecting to Ollama API after {self.max_retries} attempts")
                        except Exception as e:
                            if attempt < self.max_retries - 1:
                                await asyncio.sleep(self.retry_delay)
                                continue
                            raise RuntimeError(f"Error connecting to Ollama API: {str(e)}")
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
                            raise httpx.HTTPStatusError(
                                f"Ollama API returned status {response.status_code}",
                                request=response.request,
                                response=response
                            )
                            
                        response_data = response.json()
                        return response_data.get("response", "")
                        
                    except (httpx.ConnectTimeout, httpx.ReadTimeout) as e:
                        if attempt < self.max_retries - 1:
                            await asyncio.sleep(self.retry_delay)
                            continue
                        raise httpx.TimeoutException(f"Timeout connecting to Ollama API after {self.max_retries} attempts: {str(e)}")
                        
                    except Exception as e:
                        if attempt < self.max_retries - 1:
                            await asyncio.sleep(self.retry_delay)
                            continue
                        raise RuntimeError(f"Error connecting to Ollama API: {str(e)}")
                
                raise RuntimeError("Failed to get response from Ollama API after all retry attempts")
    
    async def generate_structured_output(
        self, 
        prompt: Union[str, List[Dict[str, Any]]], 
        schema: Union[Dict[str, Any], Type[BaseModel]], 
        max_tokens: Optional[int] = None,
        max_completion_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None,
        stream: Optional[bool] = None,
        logit_bias: Optional[Dict[str, float]] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        n: Optional[int] = None,
        seed: Optional[int] = None,
        user: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        parallel_tool_calls: Optional[bool] = None,
        response_format: Optional[Dict[str, Any]] = None,
        service_tier: Optional[str] = None,
        store: Optional[bool] = None,
        metadata: Optional[Dict[str, str]] = None,
        reasoning_effort: Optional[str] = None,
        modalities: Optional[List[str]] = None,
        audio: Optional[Dict[str, Any]] = None,
        prediction: Optional[Dict[str, Any]] = None,
        web_search_options: Optional[Dict[str, Any]] = None,
        stream_options: Optional[Dict[str, Any]] = None,
        num_ctx: Optional[int] = None,
        repeat_last_n: Optional[int] = None,
        repeat_penalty: Optional[float] = None,
        top_k: Optional[int] = None,
        min_p: Optional[float] = None,
        keep_alive: Optional[str] = None,
        think: Optional[bool] = None,
        **kwargs
    ) -> Dict[str, Any]:
        stream = stream if stream is not None else self.stream
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        max_completion_tokens = max_completion_tokens if max_completion_tokens is not None else self.max_completion_tokens
        temperature = temperature if temperature is not None else self.temperature
        top_p = top_p if top_p is not None else self.top_p
        stop = stop if stop is not None else self.stop
        logit_bias = logit_bias if logit_bias is not None else self.logit_bias
        logprobs = logprobs if logprobs is not None else self.logprobs
        top_logprobs = top_logprobs if top_logprobs is not None else self.top_logprobs
        n = n if n is not None else self.n
        seed = seed if seed is not None else self.seed
        user = user if user is not None else self.user
        tools = tools if tools is not None else self.tools
        tool_choice = tool_choice if tool_choice is not None else self.tool_choice
        parallel_tool_calls = parallel_tool_calls if parallel_tool_calls is not None else self.parallel_tool_calls
        response_format = response_format if response_format is not None else self.response_format
        service_tier = service_tier if service_tier is not None else self.service_tier
        store = store if store is not None else self.store
        metadata = metadata if metadata is not None else self.metadata
        reasoning_effort = reasoning_effort if reasoning_effort is not None else self.reasoning_effort
        modalities = modalities if modalities is not None else self.modalities
        audio = audio if audio is not None else self.audio
        prediction = prediction if prediction is not None else self.prediction
        web_search_options = web_search_options if web_search_options is not None else self.web_search_options
        stream_options = stream_options if stream_options is not None else self.stream_options
        num_ctx = num_ctx if num_ctx is not None else self.num_ctx
        repeat_last_n = repeat_last_n if repeat_last_n is not None else self.repeat_last_n
        repeat_penalty = repeat_penalty if repeat_penalty is not None else self.repeat_penalty
        top_k = top_k if top_k is not None else self.top_k
        min_p = min_p if min_p is not None else self.min_p
        keep_alive = keep_alive if keep_alive is not None else self.keep_alive
        think = think if think is not None else self.think
        
        if stream:
            self.logger.warning("Streaming is not supported for structured JSON output with Ollama, using non-streaming.")
        
        is_pydantic_schema = isinstance(schema, type) and issubclass(schema, BaseModel)
        json_schema_def = schema.model_json_schema() if is_pydantic_schema else schema
            
        options = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repeat_penalty": repeat_penalty
        }
        
        if num_ctx:
            options["num_ctx"] = num_ctx
        if repeat_last_n:
            options["repeat_last_n"] = repeat_last_n
        if min_p:
            options["min_p"] = min_p
        if seed:
            options["seed"] = seed
        if keep_alive:
            options["keep_alive"] = keep_alive
        if think:
            options["think"] = think
            
        if max_tokens or max_completion_tokens:
            options["num_predict"] = max_tokens if max_tokens else max_completion_tokens
            
        if isinstance(prompt, list):
            serializable_messages = []
            for message in prompt:
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
            
            generate_payload = {
                "model": self.model_name,
                "messages": serializable_messages,
                "options": options,
                "format": json_schema_def,
                "stream": False
            }
        else:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            generate_payload = {
                "model": self.model_name,
                "messages": messages,
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
                    self.logger.debug(f"Attempting structured output via /api/chat (attempt {attempt+1}/{self.max_retries})")
                    response = await client.post(
                        f"{self.api_base_url}/api/chat",
                        json=generate_payload,
                        timeout=7200.0
                    )
                    last_response = response

                    if response.status_code == 200:
                        response_data = response.json()
                        break
                    else:
                        last_error = f"API returned status {response.status_code}"
                        self.logger.warning(f"/api/chat failed with status {response.status_code}. Response: {response.text}")

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
            content = response_data.get("message", {}).get("content", "") or response_data.get("response", "")
            content = content.strip()

            structured_data = None
            parse_error = None
            structured_data = json.loads(content)
            if structured_data is None:
                self.logger.error(f"Failed to parse response as JSON. Error: {parse_error}")
                return {
                    "error": f"Failed to parse JSON: {parse_error}",
                    "raw_content": content
                }
            
            if is_pydantic_schema:
                try:
                    validated_data = schema.model_validate(structured_data)
                    return validated_data
                except Exception as validation_error:
                    self.logger.error(f"Schema validation failed: {validation_error}")
                    return {
                        "error": f"Schema validation failed: {validation_error}",
                        "parsed_data": structured_data
                    }
            else:
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
                raise httpx.HTTPStatusError(
                    f"Failed to get models, status: {response.status_code}",
                    request=response.request,
                    response=response
                )
        except Exception as e:
            self.logger.error(f"Error listing available models: {str(e)}")
            raise RuntimeError(f"Error listing available models: {str(e)}")