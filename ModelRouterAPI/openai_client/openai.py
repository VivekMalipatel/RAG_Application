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
        model_name: str = "gpt-4o-mini",
        system_prompt: Optional[str] = "You are a helpful AI assistant.",
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: Optional[Union[str, List[str]]] = None,
        **kwargs
    ):
        self.client = AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL
        )
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.stream = stream
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.stop = stop
        self.logger = logging.getLogger(__name__)

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
        
        if isinstance(prompt, list):
            messages = []
            for msg in prompt:
                message_copy = dict(msg)
                if message_copy.get("content") is None:
                    message_copy["content"] = ""
                messages.append(message_copy)
                
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
            "stream": stream
        }
        
        if max_tokens:
            params["max_tokens"] = max_tokens
            
        if stop:
            params["stop"] = stop
        
        if stream:
            return self._generate_stream(params)
        else:
            try:
                response = await self.client.chat.completions.create(**params)
                return response.choices[0].message.content or ""
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
        prompt: str, 
        schema: Union[Dict[str, Any], Type[BaseModel]], 
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        if stream:
            self.logger.warning("Streaming not supported for structured outputs")
            
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            json_schema = schema.model_json_schema()
        else:
            json_schema = schema
            
        try:
            system_prompt = f"{self.system_prompt}\nYou must respond with valid JSON conforming to the provided schema."
            
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response_schema",
                    "schema": json_schema
                }
            }
            
            params = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "temperature": self.temperature,
                "top_p": self.top_p,
                "response_format": response_format
            }
            
            if max_tokens:
                params["max_tokens"] = max_tokens
                
            response = await self.client.chat.completions.create(**params)
            content = response.choices[0].message.content
            
            try:
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].strip()
                    
                parsed_json = json.loads(content)
                
                if isinstance(schema, type) and issubclass(schema, BaseModel):
                    validated_data = schema.model_validate(parsed_json)
                    return validated_data.model_dump()
                
                return parsed_json
                
            except (json.JSONDecodeError, IndexError) as e:
                self.logger.error(f"Failed to parse JSON response: {str(e)}")
                return {"error": "Invalid JSON response", "content": content}
                
        except Exception as e:
            self.logger.error(f"Error with OpenAI API: {str(e)}")
            return {"error": str(e)}

    async def embed_text(self, texts: List[str]) -> List[List[float]]:
        try:
            embed_model = self.model_name
            if not "embedding" in self.model_name:
                embed_model = "text-embedding-3-small"
                
            response = await self.client.embeddings.create(
                model=embed_model,
                input=texts
            )
            
            return [item.embedding for item in response.data]
            
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {str(e)}")
            raise

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
            
            return [
                "gpt-4o", "gpt-4o-mini", "gpt-4", "gpt-4-turbo",
                "gpt-3.5-turbo", "text-embedding-3-small", "text-embedding-3-large"
            ]