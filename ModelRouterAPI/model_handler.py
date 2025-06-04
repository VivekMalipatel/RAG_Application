import logging
from typing import Optional, Dict, List, Union, AsyncGenerator, Any, Type, Tuple
from pydantic import BaseModel

from openai_client import OpenAIClient
from ollama import OllamaClient
from huggingface import HuggingFaceClient

from model_provider import Provider
from model_type import ModelType
from core.model_selector import ModelSelector
from core.model_selector import ModelNotFoundException

class UnsupportedFeatureError(Exception):
    pass

class ModelRouter:
    def __init__(
        self,
        provider: Union[Provider, str],
        model_name: str,
        model_type: ModelType,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: Optional[int] = None,
        max_completion_tokens: Optional[int] = None,
        stream: bool = False,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: Optional[Union[str, List[str]]] = None,
        logit_bias: Optional[Dict[str, float]] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        n: Optional[int] = 1,
        seed: Optional[int] = None,
        user: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        parallel_tool_calls: Optional[bool] = True,
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
        if isinstance(provider, str):
            try:
                self.provider = Provider(provider)
            except ValueError:
                raise ValueError(f"Unsupported provider: {provider}")
        else:
            self.provider = provider
            
        self.model_name = model_name
        self.model_type = model_type
        self.system_prompt = system_prompt
        self.stream = stream
        self.max_tokens = max_tokens
        self.logger = logging.getLogger(__name__)
        
        common_params = {
            "system_prompt": system_prompt,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "max_completion_tokens": max_completion_tokens,
            "stream": stream,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "stop": stop,
            "logit_bias": logit_bias,
            "logprobs": logprobs,
            "top_logprobs": top_logprobs,
            "n": n,
            "seed": seed,
            "user": user,
            "tools": tools,
            "tool_choice": tool_choice,
            "parallel_tool_calls": parallel_tool_calls,
            "response_format": response_format,
            "service_tier": service_tier,
            "store": store,
            "metadata": metadata,
            "reasoning_effort": reasoning_effort,
            "modalities": modalities,
            "audio": audio,
            "prediction": prediction,
            "web_search_options": web_search_options,
            "stream_options": stream_options,
            "num_ctx": num_ctx,
            "repeat_last_n": repeat_last_n,
            "repeat_penalty": repeat_penalty,
            "top_k": top_k,
            "min_p": min_p,
            "keep_alive": keep_alive,
            "think": think,
            **kwargs
        }
        
        if self.provider == Provider.OPENAI:
            provider_config = kwargs.get('provider_config')
            if provider_config:
                common_params['api_key'] = provider_config.get('api_key')
                common_params['base_url'] = provider_config.get('base_url')
            self.client = OpenAIClient(model_name=model_name, **common_params)
            
        elif self.provider == Provider.OLLAMA:
            self.client = OllamaClient(hf_repo=model_name, **common_params)
            
        elif self.provider == Provider.HUGGINGFACE:
            self.client = HuggingFaceClient(model_name=model_name, model_type=model_type, **common_params)
            
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    @staticmethod
    async def initialize_from_model_name(
        model_name: str,
        model_type: ModelType,
        **kwargs
    ) -> 'ModelRouter':
        selector = ModelSelector()
        
        try:
            provider, actual_model_name, provider_config = await selector.select_best_model(
                model_type=model_type, 
                model_name=model_name
            )
            
            if provider_config:
                kwargs['provider_config'] = provider_config
            
            return ModelRouter(
                provider=provider,
                model_name=actual_model_name,
                model_type=model_type,
                **kwargs
            )
        except ModelNotFoundException as e:
            raise ValueError(f"Model not found: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error initializing model router: {str(e)}")

    async def generate_text(
        self, 
        prompt: Union[str, List], 
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
        try:
            return await self.client.generate_text(
                prompt, 
                max_tokens=max_tokens,
                max_completion_tokens=max_completion_tokens,
                temperature=temperature, 
                top_p=top_p,
                stop=stop,
                stream=stream if stream is not None else self.stream,
                logit_bias=logit_bias,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
                n=n,
                seed=seed,
                user=user,
                tools=tools,
                tool_choice=tool_choice,
                parallel_tool_calls=parallel_tool_calls,
                response_format=response_format,
                service_tier=service_tier,
                store=store,
                metadata=metadata,
                reasoning_effort=reasoning_effort,
                modalities=modalities,
                audio=audio,
                prediction=prediction,
                web_search_options=web_search_options,
                stream_options=stream_options,
                num_ctx=num_ctx,
                repeat_last_n=repeat_last_n,
                repeat_penalty=repeat_penalty,
                top_k=top_k,
                min_p=min_p,
                keep_alive=keep_alive,
                think=think
            )
        except Exception as e:
            self.logger.error(f"Error generating text with {self.provider.value}: {str(e)}")
            raise

    async def embed_text(self, texts: List[str]) -> List[List[float]]:
        if not hasattr(self.client, "embed_text"):
            raise UnsupportedFeatureError(f"Provider {self.provider} does not support text embedding")
            
        try:
            return await self.client.embed_text(texts)
        except Exception as e:
            self.logger.error(f"Error generating embeddings with {self.provider.value}: {str(e)}")
            raise
    
    async def embed_image(self, image: List[dict]) -> List[List[float]]:
        if not hasattr(self.client, "embed_image"):
            raise UnsupportedFeatureError(f"Provider {self.provider} does not support image embedding")
            
        try:
            return await self.client.embed_image(image)
        except Exception as e:
            self.logger.error(f"Error generating image embeddings with {self.provider.value}: {str(e)}")
            raise

    async def rerank_documents(
        self, 
        query: str, 
        documents: List[str], 
        max_documents: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        if not hasattr(self.client, "rerank_documents"):
            raise UnsupportedFeatureError(f"Provider {self.provider} does not support document reranking")
            
        try:
            return await self.client.rerank_documents(query, documents, max_documents)
        except Exception as e:
            self.logger.error(f"Error reranking documents with {self.provider.value}: {str(e)}")
            raise

    async def generate_structured_output(
        self,
        prompt: Union[str, List],
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
        if not hasattr(self.client, "generate_structured_output"):
            raise UnsupportedFeatureError(f"Provider {self.provider} does not support structured output generation")
            
        try:
            return await self.client.generate_structured_output(
                prompt=prompt,
                schema=schema,
                max_tokens=max_tokens,
                max_completion_tokens=max_completion_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                stream=stream,
                logit_bias=logit_bias,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
                n=n,
                seed=seed,
                user=user,
                tools=tools,
                tool_choice=tool_choice,
                parallel_tool_calls=parallel_tool_calls,
                response_format=response_format,
                service_tier=service_tier,
                store=store,
                metadata=metadata,
                reasoning_effort=reasoning_effort,
                modalities=modalities,
                audio=audio,
                prediction=prediction,
                web_search_options=web_search_options,
                stream_options=stream_options,
                num_ctx=num_ctx,
                repeat_last_n=repeat_last_n,
                repeat_penalty=repeat_penalty,
                top_k=top_k,
                min_p=min_p,
                keep_alive=keep_alive,
                think=think,
                **kwargs
            )
        except Exception as e:
            self.logger.error(f"Error generating structured output with {self.provider.value}: {str(e)}")
            raise

    def set_system_prompt(self, system_prompt: str) -> None:
        self.system_prompt = system_prompt
        
        if hasattr(self.client, "set_system_prompt"):
            self.client.set_system_prompt(system_prompt)
            
    def is_model_available(self) -> bool:
        if not hasattr(self.client, "is_model_available"):
            return True
            
        try:
            return self.client.is_model_available()
        except Exception as e:
            self.logger.warning(f"Could not check model availability: {str(e)}")
            return False