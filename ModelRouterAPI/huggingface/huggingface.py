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

class HuggingFaceClient:
    """
    Unified client for interacting with Hugging Face models locally.
    Supports:
    - Text embeddings
    - Text generation
    - Document reranking
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
        device: Optional[str] = None,
        trust_remote_code: Optional[bool] = True,
        **kwargs
    ):
        """
        Initializes the Hugging Face client with local model loading.
        
        Args:
            model_name: Full model name (e.g., "mistralai/Mistral-7B-Instruct-v0.1")
            model_type: Type of model (text generation, embedding, reranker)
            system_prompt: System prompt for chat models
            temperature: Temperature parameter for sampling
            top_p: Nucleus sampling parameter
            max_tokens: Maximum tokens to generate
            stream: Whether to stream responses
            device: Device to run inference on ('cuda', 'cpu', 'mps')
            trust_remote_code: Whether to trust remote code in model repos
        """
        self.model_name = model_name
        self.model_type = model_type
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.stream = stream
        self.trust_remote_code = trust_remote_code
        
        # Extract additional parameters
        self.repetition_penalty = kwargs.get("repetition_penalty", 1.0)
        self.top_k = kwargs.get("top_k", 50)
        self.num_beams = kwargs.get("num_beams", 1)
        
        # Determine device
        self.device = (
            device
            if device
            else ("cuda" if torch.cuda.is_available() 
                  else "mps" if torch.backends.mps.is_available() 
                  else "cpu")
        )
        
        self.hf_token = settings.HUGGINGFACE_API_TOKEN
        
        self.model = None
        self.tokenizer = None
        self.processor = None
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initializing Hugging Face client with model {model_name} on {self.device}")
        
        # Ensure the model is available before proceeding
        if not self.is_model_available():
            raise ValueError(f"Model {self.model_name} is not available on Hugging Face")
            
        try:
            kwargs = {
                "trust_remote_code": self.trust_remote_code,
                "revision": "main",
                "token": self.hf_token,
                "device_map": self.device
            }
            
            if model_type == ModelType.TEXT_GENERATION:
                self._load_text_model(**kwargs)
            elif model_type == ModelType.TEXT_EMBEDDING:
                self._load_text_model(**kwargs)
            elif model_type == ModelType.RERANKER:
                self._load_reranker_model(**kwargs)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
        except Exception as e:
            self.logger.error(f"Failed to load Hugging Face model {model_name}: {str(e)}")
            raise RuntimeError(f"Model initialization failed: {model_name}")
    
    def _load_text_model(self, **kwargs):
        """Loads a text generation or embedding model."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=self.hf_token)
            self.model = AutoModel.from_pretrained(self.model_name, **kwargs)
            self.model.eval()
            self.model.to(self.device)
            self.logger.info(f"Loaded text model: {self.model_name} on {self.device}")
        except Exception as e:
            self.logger.error(f"Text model loading failed: {str(e)}")
            raise
    
    def _load_reranker_model(self, **kwargs):
        """Loads a reranker model for document ranking."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=self.hf_token)
            self.model = AutoModel.from_pretrained(self.model_name, **kwargs)
            self.model.eval()
            self.model.to(self.device)
            self.logger.info(f"Loaded reranker model: {self.model_name} on {self.device}")
        except Exception as e:
            self.logger.error(f"Reranker model loading failed: {str(e)}")
            raise
    
    def is_model_available(self) -> bool:
        """Check if the model is available on Hugging Face."""
        try:
            url = f"https://huggingface.co/api/models/{self.model_name}"
            headers = {"Authorization": f"Bearer {self.hf_token}"} if self.hf_token else {}
            response = requests.get(url, headers=headers)
            return response.status_code == 200
        except Exception as e:
            self.logger.error(f"Failed to verify model availability: {str(e)}")
            return False
    
    async def generate_text(
        self, 
        prompt: Union[str, List],
        max_tokens: Optional[int] = None,
        stream: Optional[bool] = None
    ) -> Union[str, AsyncGenerator[str, None]]:
        """
        Generate text using local Hugging Face model.
        
        Args:
            prompt: Text prompt or list of messages to generate response for
            max_tokens: Maximum number of tokens to generate
            stream: Whether to enable streaming
            
        Returns:
            Either the generated text or an AsyncGenerator yielding text chunks
        """
        stream = stream if stream is not None else self.stream
        max_length = max_tokens if max_tokens else self.max_tokens
        
        # Format prompt properly
        if isinstance(prompt, list):  # Handle message format
            full_prompt = self._format_chat_messages(prompt)
        else:
            full_prompt = self._format_chat_prompt(prompt)
        
        # Tokenize input
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
        
        if stream:
            # Use TextIteratorStreamer for proper streaming
            streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)
            generation_kwargs = {
                **inputs,
                "max_length": max_length if max_length else 1024,  # Default if not specified
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "repetition_penalty": self.repetition_penalty,
                "num_beams": self.num_beams,
                "streamer": streamer
            }
            
            # Run generation in a separate thread
            threading.Thread(target=self.model.generate, kwargs=generation_kwargs).start()
            
            async def stream_generator():
                for text in streamer:
                    yield text
            
            return stream_generator()
        else:
            try:
                # Generate text without streaming
                with torch.no_grad():
                    output_ids = self.model.generate(
                        **inputs,
                        max_length=max_length if max_length else 1024,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        top_k=self.top_k,
                        repetition_penalty=self.repetition_penalty,
                        num_beams=self.num_beams,
                    )
                
                # Decode output
                generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                
                # Remove the prompt part if needed
                if generated_text.startswith(full_prompt):
                    generated_text = generated_text[len(full_prompt):]
                
                return generated_text.strip()
            except Exception as e:
                self.logger.error(f"Error generating text: {str(e)}")
                return f"Error: {str(e)}"
    
    async def embed_text(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for text using local Hugging Face model.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            # Tokenize all texts
            inputs = self.tokenizer(texts, padding=True, return_tensors="pt").to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # Most embedding models use mean pooling over the last hidden state
                if hasattr(outputs, "last_hidden_state"):
                    embeddings = outputs.last_hidden_state.mean(dim=1)
                else:
                    # Fallback for models with different output structure
                    embeddings = outputs[0].mean(dim=1)
            
            # Convert to Python list
            return embeddings.cpu().numpy().tolist()
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {str(e)}")
            # Return zero embeddings as fallback
            return [[0.0] * 768 for _ in range(len(texts))]  # Common embedding size
    
    async def rerank_documents(self, query: str, documents: List[str], max_tokens: int = 512) -> List[int]:
        """
        Reranks documents based on query relevance using a reranker model.
        
        Args:
            query: The search query
            documents: List of document texts to rank
            max_tokens: Maximum tokens to consider for long documents
            
        Returns:
            List of document indices sorted by relevance (highest first)
        """
        try:
            query_tokens = self.tokenizer(query, return_tensors="pt").to(self.device)
            doc_tokens = self.tokenizer(documents, return_tensors="pt", padding=True, truncation=True, max_length=max_tokens).to(self.device)
            
            # Handle long documents by truncating if needed
            if any(len(doc) > max_tokens for doc in documents):
                truncated_documents = []
                for doc in documents:
                    truncated_documents.append(doc[:max_tokens*4] + ".....")  # Rough conversion from tokens to chars
                documents = truncated_documents
                doc_tokens = self.tokenizer(documents, return_tensors="pt", padding=True, truncation=True, max_length=max_tokens).to(self.device)
            
            with torch.no_grad():
                # Get query embedding
                query_embedding = self.model(**query_tokens).last_hidden_state.mean(dim=1)
                
                # Get document embeddings
                doc_embeddings = self.model(**doc_tokens).last_hidden_state.mean(dim=1)
                
                # Compute relevance scores by dot product
                scores = torch.matmul(query_embedding, doc_embeddings.T).squeeze().cpu().numpy()
            
            # Return indices sorted by relevance (highest first)
            return np.argsort(scores)[::-1].tolist()
        except Exception as e:
            self.logger.error(f"Error reranking documents: {str(e)}")
            # Return original order as fallback
            return list(range(len(documents)))
    
    def _format_chat_messages(self, messages: List[Dict]) -> str:
        """Convert a list of message dictionaries to a single formatted prompt string."""
        formatted = ""
        model_name_lower = self.model_name.lower()
        
        # Apply model-specific formatting
        if "mistral" in model_name_lower and "instruct" in model_name_lower:
            # Mistral formatting
            system = next((msg["content"] for msg in messages if msg["role"] == "system"), None)
            formatted = "<s>"
            for msg in messages:
                if msg["role"] == "system":
                    continue  # Handled separately
                elif msg["role"] == "user":
                    if system and formatted == "<s>":
                        formatted += f"[INST] {system}\n\n{msg['content']} [/INST]"
                    else:
                        formatted += f"[INST] {msg['content']} [/INST]"
                elif msg["role"] == "assistant":
                    formatted += f" {msg['content']} </s>"
            
        elif "llama" in model_name_lower:
            # Llama formatting 
            system = next((msg["content"] for msg in messages if msg["role"] == "system"), None)
            formatted = "<s>"
            for msg in messages:
                if msg["role"] == "system":
                    continue  # Handled separately
                elif msg["role"] == "user":
                    if system and formatted == "<s>":
                        formatted += f"[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{msg['content']} [/INST]"
                    else:
                        formatted += f"[INST] {msg['content']} [/INST]"
                elif msg["role"] == "assistant":
                    formatted += f" {msg['content']} </s>"
        
        else:
            # Generic formatting
            for msg in messages:
                role_prefix = {
                    "system": "System: ",
                    "user": "User: ",
                    "assistant": "Assistant: ",
                    "function": "Function: ",
                }.get(msg["role"], f"{msg['role'].capitalize()}: ")
                formatted += f"{role_prefix}{msg['content']}\n"
            formatted += "Assistant: "
            
        return formatted
    
    def _format_chat_prompt(self, prompt: str) -> str:
        """
        Format a single text prompt according to the model's expected format.
        
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
        self.logger.info(f"System prompt updated for {self.model_name}: {system_prompt}")