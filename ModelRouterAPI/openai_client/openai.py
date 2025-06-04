import logging
import json
from typing import Optional, List, Union, AsyncGenerator, Dict, Any, Type
from pydantic import BaseModel
from openai import AsyncOpenAI, APIError
from config import settings
import asyncio
import nest_asyncio

class OpenAIClient:
    def __init__(
        self,
        model_name: str = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
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
        **kwargs
    ):
        self.client = AsyncOpenAI(
            api_key=api_key or settings.OPENAI_API_KEY,
            base_url=base_url or settings.OPENAI_BASE_URL
        )
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.max_completion_tokens = max_completion_tokens
        self.stream = stream
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.stop = stop
        self.logit_bias = logit_bias
        self.logprobs = logprobs
        self.top_logprobs = top_logprobs
        self.n = n
        self.seed = seed
        self.user = user
        self.tools = tools
        self.tool_choice = tool_choice
        self.parallel_tool_calls = parallel_tool_calls
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
        self.logger = logging.getLogger(__name__)

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
        **kwargs: Any
    ) -> Union[str, List[str], AsyncGenerator[Dict[str, Any], None]]:
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
        
        if isinstance(prompt, list):
            messages = prompt.copy()
            if not any(m.role == "system" for m in messages):
                messages.insert(0, {"role": "system", "content": self.system_prompt or ""})
        else:
            messages = [
                {"role": "system", "content": self.system_prompt or ""},
                {"role": "user", "content": prompt}
            ]
        
        params = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "stream": stream
        }
        
        if max_tokens:
            params["max_tokens"] = max_tokens
            
        if max_completion_tokens:
            params["max_completion_tokens"] = max_completion_tokens
            
        if stop:
            params["stop"] = stop
            
        if logit_bias:
            params["logit_bias"] = logit_bias
            
        if logprobs is not None:
            params["logprobs"] = logprobs
            
        if top_logprobs:
            params["top_logprobs"] = top_logprobs
            
        if n and n != 1:
            params["n"] = n
            
        if seed:
            params["seed"] = seed
            
        if user:
            params["user"] = user
            
        if tools:
            params["tools"] = tools
            
        if tool_choice:
            params["tool_choice"] = tool_choice
            
        if parallel_tool_calls is not None:
            params["parallel_tool_calls"] = parallel_tool_calls
            
        if response_format:
            params["response_format"] = response_format
            
        if service_tier:
            params["service_tier"] = service_tier
            
        if store is not None:
            params["store"] = store
            
        if metadata:
            params["metadata"] = metadata
            
        if reasoning_effort:
            params["reasoning_effort"] = reasoning_effort
            
        if modalities:
            params["modalities"] = modalities
            
        if audio:
            params["audio"] = audio
            
        if prediction:
            params["prediction"] = prediction
            
        if web_search_options:
            params["web_search_options"] = web_search_options
            
        if stream_options and stream:
            params["stream_options"] = stream_options
        
        if stream:
            return self._generate_stream(params)
        else:
            try:
                response = await self.client.chat.completions.create(**params)
                print(response)
                if n and n > 1:
                    return [choice.message for choice in response.choices]
                return response.choices[0].message
            except Exception as e:
                self.logger.error(f"Error with OpenAI API: {str(e)}")
                raise
                
    async def _generate_stream(self, params: Dict[str, Any]) -> AsyncGenerator[str, None]:
        try:
            stream = await self.client.chat.completions.create(**params)
            async for chunk in stream:
                content = ""
                if hasattr(chunk.choices[0].delta, "content") and chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                yield content
        except Exception as e:
            self.logger.error(f"Error in streaming: {str(e)}")
            yield ""

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
        service_tier: Optional[str] = None,
        store: Optional[bool] = None,
        metadata: Optional[Dict[str, str]] = None,
        reasoning_effort: Optional[str] = None,
        modalities: Optional[List[str]] = None,
        audio: Optional[Dict[str, Any]] = None,
        prediction: Optional[Dict[str, Any]] = None,
        web_search_options: Optional[Dict[str, Any]] = None,
        stream_options: Optional[Dict[str, Any]] = None,
        **kwargs: Any
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
        service_tier = service_tier if service_tier is not None else self.service_tier
        store = store if store is not None else self.store
        metadata = metadata if metadata is not None else self.metadata
        reasoning_effort = reasoning_effort if reasoning_effort is not None else self.reasoning_effort
        modalities = modalities if modalities is not None else self.modalities
        audio = audio if audio is not None else self.audio
        prediction = prediction if prediction is not None else self.prediction
        web_search_options = web_search_options if web_search_options is not None else self.web_search_options
        stream_options = stream_options if stream_options is not None else self.stream_options
        
        if stream:
            self.logger.warning("Streaming not supported for structured outputs")
            
        try:
            if isinstance(prompt, list):
                messages = prompt.copy()
                if not any(m.get("role") == "system" for m in messages):
                    messages.insert(0, {"role": "system", "content": self.system_prompt})
            else:
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ]
            
            params = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "frequency_penalty": self.frequency_penalty,
                "presence_penalty": self.presence_penalty,
            }
            
            if max_tokens:
                params["max_tokens"] = max_tokens
                
            if max_completion_tokens:
                params["max_completion_tokens"] = max_completion_tokens
                
            if stop:
                params["stop"] = stop
                
            if logit_bias:
                params["logit_bias"] = logit_bias
                
            if logprobs is not None:
                params["logprobs"] = logprobs
                
            if top_logprobs:
                params["top_logprobs"] = top_logprobs
                
            if n and n != 1:
                params["n"] = n
                
            if seed:
                params["seed"] = seed
                
            if user:
                params["user"] = user
                
            if tools:
                params["tools"] = tools
                
            if tool_choice:
                params["tool_choice"] = tool_choice
                
            if parallel_tool_calls is not None:
                params["parallel_tool_calls"] = parallel_tool_calls
                
            if service_tier:
                params["service_tier"] = service_tier
                
            if store is not None:
                params["store"] = store
                
            if metadata:
                params["metadata"] = metadata
                
            if reasoning_effort:
                params["reasoning_effort"] = reasoning_effort
                
            if modalities:
                params["modalities"] = modalities
                
            if audio:
                params["audio"] = audio
                
            if prediction:
                params["prediction"] = prediction
                
            if web_search_options:
                params["web_search_options"] = web_search_options
                
            if stream_options and not stream:
                params["stream_options"] = stream_options
            
            if isinstance(schema, type) and issubclass(schema, BaseModel):
                response = await self.client.beta.chat.completions.parse(
                    response_format=schema,
                    **params
                )
                return response.choices[0].message.parsed
            else:
                response_format = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "response_schema",
                        "schema": schema
                    }
                }
                params["response_format"] = response_format
                
                response = await self.client.chat.completions.create(**params)
                content = response.choices[0].message.content
                
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].strip()
                    
                return json.loads(content)
                
        except Exception as e:
            self.logger.error(f"Error with OpenAI API: {str(e)}")
            return {"error": str(e)}

    def is_model_available(self) -> bool:
        nest_asyncio.apply()
        
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            model_list = loop.run_until_complete(self.get_model_list())
            loop.close()
            
            return self.model_name in model_list
        except Exception as e:
            self.logger.error(f"Error checking model availability: {str(e)}")
            
            known_patterns = [
                "gpt-4", "gpt-3.5", "text-embedding", "dall-e", 
                "whisper", "claude", "command", "llama"
            ]
            return any(pattern in self.model_name.lower() for pattern in known_patterns)

    def set_system_prompt(self, system_prompt: str):
        self.system_prompt = system_prompt

    async def get_model_list(self) -> List[str]:
        try:
            response = await self.client.models.list()
            
            model_ids = [model.id for model in response.data]
                
            return model_ids
            
        except Exception as e:
            self.logger.error(f"Error fetching model list from OpenAI: {str(e)}")