import torch
import logging
from transformers import AutoTokenizer, AutoModel,TextIteratorStreamer
from typing import List, Union, Optional, AsyncGenerator
import threading
import requests
import numpy as np
from app.config import settings
from app.core.models.model_type import ModelType

class HuggingFaceClient:
    """
    Unified client for interacting with Hugging Face models.
    Supports:
    - Text embeddings (Nomic, BGE, etc.)
    - Vision embeddings (CLIP, Nomic, etc.)
    - Text generation (LLaMA, Mistral, etc.)
    - Image generation (Stable Diffusion, SDXL, etc.)
    - Document reranking (JinaAI ColBERT v2)
    """

    def __init__(
        self,
        model_name: str,
        model_type: ModelType,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = 0.7,
        top_p: Optional[float] = 0.9,
        max_tokens: Optional[int] = None,
        stream: Optional[bool] = False,
        device: Optional[str] = None,
        trust_remote_code: Optional[bool] = True
    ):
        """
        Initializes a Hugging Face model client.

        Args:
            model_name (str): Hugging Face model repository ID.
            model_type (str): One of ['text', 'image', 'generation', 'image-generation', 'reranker'].
            system_prompt (Optional[str]): Custom system prompt for the model.
            temperature (float): Sampling temperature for randomness.
            top_p (float): Probability mass for nucleus sampling.
            max_tokens (int): Maximum tokens allowed in response.
            top_k (int): Number of top candidates considered.
            num_beams (int): Beam search width.
            stream (bool): Whether to enable streaming responses.
            device (Optional[str]): Device to run inference ('cuda', 'cpu', 'mps'). Defaults to best available.
            trust_remote_code (bool): Whether to trust remote code execution.
        """
        self.device = (
            device
            if device
            else ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        )
        self.model_name = model_name
        self.model_type = model_type
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.stream = stream
        self.trust_remote_code = trust_remote_code
        self.hf_token = settings.HUGGINGFACE_TOKEN  # Hugging Face token for authenticated models

        self.model = None
        self.tokenizer = None
        self.processor = None

        # Ensure the model is available before proceeding
        if not self.is_model_available():
            raise ValueError(f"Model {self.model_name} is not available on Hugging Face.")

        try:
            kwargs = {
                "trust_remote_code": self.trust_remote_code,
                "revision": "main",
                "token": self.hf_token,  # Use token if required
                "device_map": self.device
            }

            if model_type == ModelType.TEXT_GENERATION or model_type == ModelType.TEXT_EMBEDDING:
                self._load_text_model(**kwargs)
            elif model_type == ModelType.RERANKER:
                self._load_reranker_model(**kwargs)
            else:
                raise ValueError(f"Unsupported model task: {model_type}")
        except Exception as e:
            logging.error(f"Failed to load Hugging Face model {model_name}: {str(e)}")
            raise RuntimeError(f"Model initialization failed: {model_name}")

    def _load_text_model(self, **kwargs):
        """Loads a text generation model (e.g., LLaMA, Mistral)."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=self.hf_token)
            self.model = AutoModel.from_pretrained(self.model_name, **kwargs)
            self.model.eval()
            self.model.to(self.device)
            logging.info(f"Loaded text generation model: {self.model_name} on {self.device}")
        except Exception as e:
            logging.error(f"Text generation model loading failed: {str(e)}")
            raise

    def _load_reranker_model(self, **kwargs):
        """Loads a reranker model for document ranking."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=self.hf_token)
            self.model = AutoModel.from_pretrained(self.model_name, **kwargs)
            self.model.eval()
            self.model.to(self.device)
            logging.info(f"Loaded reranker model: {self.model_name} on {self.device}")
        except Exception as e:
            logging.error(f"Reranker model loading failed: {str(e)}")
            raise

    def is_model_available(self) -> bool:
        """
        Checks if the requested model is available on Hugging Face.
        """
        try:
            url = f"https://huggingface.co/api/models/{self.model_name}"
            headers = {"Authorization": f"Bearer {self.hf_token}"} if self.hf_token else {}
            response = requests.get(url, headers=headers)
            return response.status_code == 200
        except Exception as e:
            logging.error(f"Failed to verify model availability: {str(e)}")
            return False

    async def generate_text(self, prompt: str, max_tokens: Optional[int] = None, stream: Optional[bool] = None) -> Union[str, AsyncGenerator[str, None]]:
        """Generates text using a Hugging Face text generation model with optional streaming."""
        
        full_prompt = prompt
        if self.system_prompt:
            full_prompt = f"{self.system_prompt}\n\n{prompt}"
        
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
        stream = stream if stream is not None else self.stream
        
        if self.stream:
            # Use proper HF streaming implementation
            streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)
            generation_kwargs = {
                **inputs,
                "max_length": max_tokens if max_tokens else self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "streamer": streamer
            }
            
            # Run generation in a separate thread
            threading.Thread(target=self.model.generate, kwargs=generation_kwargs).start()
            
            async def stream_generator():
                for text in streamer:
                    yield text
            return stream_generator()
        else:
            output = self.model.generate(
                **inputs,
                max_length=max_tokens if max_tokens else self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            return self.tokenizer.decode(output[0], skip_special_tokens=True)
    
    def embed_text(self, texts: List[str]) -> List[List[float]]:
        """Generates text embeddings."""
        inputs = self.tokenizer(texts, padding=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            embeddings = self.model(**inputs).last_hidden_state.mean(dim=1)
        return embeddings.cpu().numpy().tolist()

    def rerank_documents(self, query: str, documents: List[str]) -> List[int]:
        """Uses JinaAI ColBERT v2 for document reranking."""
        query_tokens = self.tokenizer(query, return_tensors="pt").to(self.device)
        doc_tokens = self.tokenizer(documents, return_tensors="pt", padding=True).to(self.device)

        with torch.no_grad():
            query_embedding = self.model(**query_tokens).last_hidden_state.mean(dim=1)
            doc_embeddings = self.model(**doc_tokens).last_hidden_state.mean(dim=1)
            scores = torch.matmul(query_embedding, doc_embeddings.T).squeeze().cpu().numpy()

        return np.argsort(scores)[::-1].tolist()

    def set_system_prompt(self, system_prompt: str):
        """Updates the system prompt dynamically."""
        self.system_prompt = system_prompt
        logging.info(f"System prompt updated for Hugging Face model {self.model_name}: {system_prompt}")