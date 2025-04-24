import logging
import asyncio
from typing import List, Dict, Any, Optional, Union, AsyncGenerator
import torch
import numpy as np
import requests
from transformers import AutoTokenizer, AutoModel, TextIteratorStreamer
import threading

from config import settings
from model_type import ModelType
from huggingface.model_cache import ModelCache
from core.device_utils import DeviceManager

class HuggingFaceClient:
    """Unified client for interacting with local Hugging Face models"""
    
    def __init__(
        self,
        model_name: str,
        model_type: ModelType,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        device: Optional[str] = None,
        trust_remote_code: bool = True,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: Optional[Union[str, List[str]]] = None,
        **kwargs
    ):
        self.model_name = model_name
        self.model_type = model_type
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.stream = stream
        self.trust_remote_code = trust_remote_code
        self.frequency_penalty = kwargs.get("frequency_penalty", frequency_penalty)
        self.presence_penalty = kwargs.get("presence_penalty", presence_penalty)
        self.stop = stop
        
        # Additional generation parameters
        self.repetition_penalty = kwargs.get("repetition_penalty", 1.1)
        self.top_k = kwargs.get("top_k", 50)
        self.num_beams = kwargs.get("num_beams", 1)
        
        # Determine device using the DeviceManager
        self.device = device if device else DeviceManager.get_optimal_device()
        
        self.hf_token = settings.HUGGINGFACE_API_TOKEN
        
        self.model = None
        self.tokenizer = None
        
        self.logger = logging.getLogger(__name__)
        
        # Load model from cache
        try:
            self._load_from_cache()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize model {model_name}: {str(e)}")
    
    def _load_from_cache(self):
        """Load model and tokenizer from the shared model cache"""
        model_cache = ModelCache()
        self.model, self.tokenizer = model_cache.get_model(
            model_name=self.model_name,
            model_type=self.model_type,
            device=self.device,
            token=self.hf_token,
            trust_remote_code=self.trust_remote_code
        )
    
    def is_model_available(self) -> bool:
        """Check if the model is available on Hugging Face"""
        try:
            url = f"https://huggingface.co/api/models/{self.model_name}"
            headers = {"Authorization": f"Bearer {self.hf_token}"} if self.hf_token else {}
            response = requests.get(url, headers=headers, timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    async def generate_text(
        self, 
        prompt: Union[str, List],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None,
        stream: Optional[bool] = None
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Generate text using the loaded model"""
        stream = self.stream if stream is None else stream
        max_length = max_tokens if max_tokens else self.max_tokens or 1024
        temperature = temperature if temperature is not None else self.temperature
        top_p = top_p if top_p is not None else self.top_p
        stop = stop if stop is not None else self.stop
        
        # Format prompt depending on the input type
        if isinstance(prompt, list):
            full_prompt = self._format_chat_messages(prompt)
        else:
            full_prompt = self._format_chat_prompt(prompt)
        
        # Tokenize input
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
        
        # Configure stop sequences if provided
        stopping_criteria = None
        if stop:
            from transformers import StoppingCriteriaList, StoppingCriteria
            
            class StopSequenceCriteria(StoppingCriteria):
                def __init__(self, stop_sequences, tokenizer):
                    super().__init__()
                    self.stop_sequences = stop_sequences if isinstance(stop_sequences, list) else [stop_sequences]
                    self.tokenizer = tokenizer
                    
                def __call__(self, input_ids, scores, **kwargs):
                    for stop_seq in self.stop_sequences:
                        if stop_seq in self.tokenizer.decode(input_ids[0][-len(self.tokenizer.encode(stop_seq, add_special_tokens=False)):]):
                            return True
                    return False
            
            stopping_criteria = StoppingCriteriaList([StopSequenceCriteria(stop, self.tokenizer)])
        
        gen_kwargs = {
            "max_length": max_length + len(inputs.input_ids[0]),  # Add prompt length
            "temperature": temperature,
            "top_p": top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "num_beams": self.num_beams,
            "do_sample": temperature > 0.0,
        }
        
        if stopping_criteria:
            gen_kwargs["stopping_criteria"] = stopping_criteria
        
        if stream:
            # Use TextIteratorStreamer for streaming
            streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True, timeout=15.0)
            gen_kwargs["streamer"] = streamer
            
            # Run generation in a separate thread
            thread = threading.Thread(target=self.model.generate, kwargs={**inputs, **gen_kwargs})
            thread.start()
            
            async def stream_generator():
                prompt_len = len(full_prompt)
                generated = ""
                for text in streamer:
                    chunk = text
                    # Skip first chunk if it's part of the prompt
                    if not generated and len(text) > 0 and text[0] == full_prompt[0]:
                        if text.startswith(full_prompt[:30]):  # Rough check for prompt prefix
                            chunk = text[prompt_len:].lstrip()
                    
                    generated += chunk
                    if chunk:
                        yield chunk
            
            return stream_generator()
        else:
            # Generate without streaming
            with torch.no_grad():
                output_ids = self.model.generate(**inputs, **gen_kwargs)
                
                # Decode output and remove the prompt part
                generated_text = self.tokenizer.decode(output_ids[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
                return generated_text.strip()
    
    async def embed_text(self, texts: List[str], dimensions: Optional[int] = None) -> List[List[float]]:
        """Generate embeddings for a list of texts with optional dimension reduction"""
        try:
            # Tokenize all texts
            inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                # Get embeddings based on model output structure
                outputs = self.model(**inputs)
                if hasattr(outputs, "pooler_output"):
                    # BERT-style models have pooler_output
                    embeddings = outputs.pooler_output
                elif hasattr(outputs, "last_hidden_state"):
                    # Use mean pooling for models without pooler
                    embeddings = outputs.last_hidden_state.mean(dim=1)
                else:
                    # Fallback for other model types
                    embeddings = outputs[0].mean(dim=1)
            
            # Normalize embeddings
            normalized = embeddings / embeddings.norm(dim=1, keepdim=True)
            
            # Reduce dimensions if requested
            if dimensions and dimensions > 0 and dimensions < normalized.shape[1]:
                # Use PCA for dimension reduction
                import torch.nn.functional as F
                
                # Move to CPU for numpy operations if needed
                cpu_embeddings = normalized.cpu().numpy()
                from sklearn.decomposition import PCA
                
                # Fit PCA and transform
                pca = PCA(n_components=dimensions)
                reduced_embeddings = pca.fit_transform(cpu_embeddings)
                
                # Normalize the reduced embeddings again
                from sklearn.preprocessing import normalize
                reduced_embeddings = normalize(reduced_embeddings)
                
                return reduced_embeddings.tolist()
            
            return normalized.cpu().numpy().tolist()
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    async def rerank_documents(
        self, 
        query: str, 
        documents: List[str],
        max_documents: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents based on query relevance.
        Returns list with {"document": text, "relevance_score": float} items.
        """
        try:
            # Cap at 512 tokens for reranker input
            query_tokens = self.tokenizer(query, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            doc_tokens = self.tokenizer(documents, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            
            with torch.no_grad():
                # For cross-encoders/rerankers
                if hasattr(self.model, "encode_query_and_docs"):
                    scores = self.model.encode_query_and_docs(query, documents)
                else:
                    # Generic approach using cosine similarity
                    query_embedding = self.model(**query_tokens).last_hidden_state.mean(dim=1)
                    doc_embeddings = self.model(**doc_tokens).last_hidden_state.mean(dim=1)
                    scores = torch.matmul(query_embedding, doc_embeddings.T).squeeze().cpu().numpy()
            
            # Create result with scores and original document text
            results = []
            for i, score in enumerate(scores):
                results.append({
                    "document": documents[i],
                    "relevance_score": float(score)
                })
            
            # Sort by score descending
            results.sort(key=lambda x: x["relevance_score"], reverse=True)
            
            # Limit results if specified
            if max_documents:
                results = results[:max_documents]
                
            return results
        except Exception as e:
            self.logger.error(f"Error reranking documents: {str(e)}")
            raise
    
    def _format_chat_messages(self, messages: List[Dict]) -> str:
        """Format messages based on model type"""
        model_name_lower = self.model_name.lower()
        
        # Extract system message if present
        system_msg = None
        for msg in messages:
            if msg.get("role") == "system":
                system_msg = msg.get("content", "")
                break
        
        # Apply model-specific formatting
        if any(x in model_name_lower for x in ["mistral", "mixtral"]) and "instruct" in model_name_lower:
            # Mistral/Mixtral Instruct format
            formatted = ""
            for i, msg in enumerate(messages):
                if msg.get("role") == "system":
                    continue  # Handled separately
                elif msg.get("role") == "user":
                    if system_msg and i == (0 if not messages[0].get("role") == "system" else 1):
                        formatted += f"<s>[INST] {system_msg}\n\n{msg.get('content', '')} [/INST]"
                    else:
                        formatted += f"<s>[INST] {msg.get('content', '')} [/INST]"
                elif msg.get("role") == "assistant":
                    formatted += f" {msg.get('content', '')} </s>"
            
            # Add assistant prefix for completion
            if formatted.endswith("</s>") or not formatted:
                formatted += "<s>[INST]"
            return formatted
            
        elif "llama" in model_name_lower:
            # Llama format
            formatted = ""
            for i, msg in enumerate(messages):
                if msg.get("role") == "system":
                    continue  # Handled separately
                elif msg.get("role") == "user":
                    if system_msg and i == (0 if not messages[0].get("role") == "system" else 1):
                        formatted += f"<s>[INST] <<SYS>>\n{system_msg}\n<</SYS>>\n\n{msg.get('content', '')} [/INST]"
                    else:
                        formatted += f"<s>[INST] {msg.get('content', '')} [/INST]"
                elif msg.get("role") == "assistant":
                    formatted += f" {msg.get('content', '')} </s>"
            
            # Add assistant prefix for completion
            if formatted.endswith("</s>") or not formatted:
                formatted += "<s>[INST]"
            return formatted
        
        else:
            # Generic chat formatting (works with most models)
            formatted = ""
            
            # Add system message if present
            if system_msg:
                formatted += f"System: {system_msg}\n\n"
                
            # Add conversation history
            for msg in messages:
                if msg.get("role") == "system":
                    continue  # Already handled
                
                role_prefix = {
                    "user": "User: ",
                    "assistant": "Assistant: ",
                    "function": "Function: ",
                }.get(msg.get("role", ""), f"{msg.get('role', '').capitalize()}: ")
                
                formatted += f"{role_prefix}{msg.get('content', '')}\n"
            
            # Add assistant prefix for the model to continue
            if not formatted.endswith("Assistant: "):
                formatted += "Assistant: "
                
            return formatted
    
    def _format_chat_prompt(self, prompt: str) -> str:
        """Format a single text prompt based on model type"""
        model_name_lower = self.model_name.lower()
        
        if any(x in model_name_lower for x in ["mistral", "mixtral"]) and "instruct" in model_name_lower:
            if self.system_prompt:
                return f"<s>[INST] {self.system_prompt}\n\n{prompt} [/INST]"
            return f"<s>[INST] {prompt} [/INST]"
                
        elif "llama" in model_name_lower:
            if self.system_prompt:
                return f"<s>[INST] <<SYS>>\n{self.system_prompt}\n<</SYS>>\n\n{prompt} [/INST]"
            return f"<s>[INST] {prompt} [/INST]"
        
        else:
            # Generic format
            if self.system_prompt:
                return f"System: {self.system_prompt}\n\nUser: {prompt}\nAssistant: "
            return f"User: {prompt}\nAssistant: "
                
    def set_system_prompt(self, system_prompt: str):
        """Update the system prompt"""
        self.system_prompt = system_prompt