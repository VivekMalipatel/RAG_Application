import torch
import logging
import asyncio
from transformers import AutoTokenizer, AutoModel, TextIteratorStreamer
from typing import List, Union, Optional, AsyncGenerator
import threading
import requests
import numpy as np
from config import settings
from model_type import ModelType
from huggingface.model_cache import ModelCache

class HuggingFaceClient:
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
        trust_remote_code: Optional[bool] = True,
        **kwargs
    ):
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
        self.hf_token = settings.HUGGINGFACE_API_TOKEN

        self.model = None
        self.tokenizer = None
        self.processor = None
        self.logger = logging.getLogger(__name__)

        if not self.is_model_available():
            raise ValueError(f"Model {self.model_name} is not available on Hugging Face.")

        try:
            kwargs = {
                "trust_remote_code": self.trust_remote_code,
                "revision": "main",
                "token": self.hf_token,
                "device_map": self.device
            }

            if model_type == ModelType.TEXT_GENERATION or model_type == ModelType.TEXT_EMBEDDING or model_type == ModelType.IMAGE_EMBEDDING:
                self._load_text_model(**kwargs)
            elif model_type == ModelType.RERANKER:
                self._load_reranker_model(**kwargs)
            else:
                raise ValueError(f"Unsupported model task: {model_type}")
        except Exception as e:
            self.logger.error(f"Failed to load Hugging Face model {model_name}: {str(e)}")
            raise RuntimeError(f"Model initialization failed: {model_name}")

    def _load_text_model(self, **kwargs):
        try:
            model_cache = ModelCache()
            
            is_nomic_multimodal = any(model_id in self.model_name for model_id in [
                "nomic-ai/colnomic-embed-multimodal",
                "nomic-ai/nomic-embed-multimodal"
            ])
            
            if is_nomic_multimodal:
                self.logger.info(f"Loading Nomic multimodal model: {self.model_name}")
                
            self.model, self.tokenizer = model_cache.get_model(
                model_name=self.model_name,
                model_type=self.model_type,
                device=self.device,
                token=self.hf_token,
                trust_remote_code=self.trust_remote_code
            )
            
            if is_nomic_multimodal:
                self.processor = self.tokenizer
                self.logger.info(f"Stored processor for Nomic multimodal model: {self.model_name}")
                
            self.logger.info(f"Loaded text model: {self.model_name} on {self.device}")
        except Exception as e:
            self.logger.error(f"Text model loading failed: {str(e)}")
            raise

    def _load_reranker_model(self, **kwargs):
        try:
            model_cache = ModelCache()
            self.model, self.tokenizer = model_cache.get_model(
                model_name=self.model_name,
                model_type=self.model_type,
                device=self.device,
                token=self.hf_token,
                trust_remote_code=self.trust_remote_code
            )
            self.logger.info(f"Loaded reranker model: {self.model_name} on {self.device}")
        except Exception as e:
            self.logger.error(f"Reranker model loading failed: {str(e)}")
            raise

    def is_model_available(self) -> bool:
        try:
            url = f"https://huggingface.co/api/models/{self.model_name}"
            headers = {"Authorization": f"Bearer {self.hf_token}"} if self.hf_token else {}
            response = requests.get(url, headers=headers)
            return response.status_code == 200
        except Exception as e:
            self.logger.error(f"Failed to verify model availability: {str(e)}")
            return False

    async def generate_text(self, prompt: str, max_tokens: Optional[int] = None, stream: Optional[bool] = None) -> Union[str, AsyncGenerator[str, None]]:
        full_prompt = prompt
        if self.system_prompt:
            full_prompt = f"{self.system_prompt}\n\n{prompt}"
        
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
        stream = stream if stream is not None else self.stream
        
        if self.stream:
            streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)
            generation_kwargs = {
                **inputs,
                "max_length": max_tokens if max_tokens else self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "streamer": streamer
            }
            
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
    
    async def embed_text(self, texts: List[str]) -> List[List[float]]:
        if "colnomic-embed-multimodal" in self.model_name or "nomic-embed-multimodal" in self.model_name:
            self.logger.info(f"Using specialized processing for Nomic multimodal model: {self.model_name}")
            try:
                batch_queries = self.processor.process_queries(texts)
                
                batch_queries = batch_queries.to(device=self.device, dtype=torch.float32 if self.device == 'cpu' else None)

                if self.device == 'cpu' and hasattr(self.model, 'to'):
                    self.model = self.model.to(dtype=torch.float32)
                
                with torch.no_grad():
                    query_embeddings = self.model(**batch_queries)
                
                return query_embeddings.cpu().numpy().tolist()
            except Exception as e:
                self.logger.error(f"Error generating embeddings with Nomic model: {e}")
                raise
        else:
            inputs = self.tokenizer(texts, padding=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                embeddings = self.model(**inputs).last_hidden_state.mean(dim=1)
            return embeddings.cpu().numpy().tolist()
    
    async def embed_image(self, images: List[dict]) -> List[List[float]]:
        if "colnomic-embed-multimodal" in self.model_name or "nomic-embed-multimodal" in self.model_name:
            self.logger.info(f"Using specialized processing for Nomic multimodal model: {self.model_name}")
            try:
                processed_images = []
                context_prompts = []
                
                import base64
                from io import BytesIO
                from PIL import Image

                max_wait_time = 30
                wait_time = 0
                check_interval = 0.5 

                while self.processor is None and wait_time < max_wait_time:
                    self.logger.warning(f"Processor not yet initialized, waiting... ({wait_time}s/{max_wait_time}s)")
                    await asyncio.sleep(check_interval)
                    wait_time += check_interval
                
                for image_data in images:
                    if isinstance(image_data["image"], str):
                        base64_str = image_data["image"]
                        img_data = base64.b64decode(base64_str)
                        img_bytes = BytesIO(img_data)
                        img = Image.open(img_bytes)
                        processed_images.append(img)
                    else:
                        raise ValueError("Images must be base64 encoded")
                    
                    context_prompts.append(image_data["text"])
                
                batch_images = self.processor.process_images(
                    images=processed_images, 
                    context_prompts=context_prompts
                )
                
                batch_images = batch_images.to(device=self.device, dtype=torch.float32 if self.device == 'cpu' else None)

                if self.device == 'cpu' and hasattr(self.model, 'to'):
                    self.model = self.model.to(dtype=torch.float32)
                
                with torch.no_grad():
                    image_embeddings = self.model(**batch_images)
                
                return image_embeddings.cpu().numpy().tolist()
            except Exception as e:
                self.logger.error(f"Error generating embeddings with Nomic model: {e}")
                raise
        else:
            self.logger.error("Image embedding is not supported for this model.")
            raise NotImplementedError("Image embedding is not supported for this model.")

    async def rerank_documents(self, query: str, documents: List[str], max_tokens: int) -> List[int]:
        query_tokens = self.tokenizer(query, return_tensors="pt").to(self.device)
        doc_tokens = self.tokenizer(documents, return_tensors="pt", padding=True).to(self.device)

        if len(doc_tokens)>8000:
            truncated_documents = []
            for doc in documents:
                truncated_documents.append(doc[:max_tokens-5]+".....")
            documents = truncated_documents
            doc_tokens = self.tokenizer(documents, return_tensors="pt", padding=True).to(self.device)

        with torch.no_grad():
            query_embedding = self.model(**query_tokens).last_hidden_state.mean(dim=1)
            doc_embeddings = self.model(**doc_tokens).last_hidden_state.mean(dim=1)
            scores = torch.matmul(query_embedding, doc_embeddings.T).squeeze().cpu().numpy()

        return np.argsort(scores)[::-1].tolist()

    def set_system_prompt(self, system_prompt: str):
        self.system_prompt = system_prompt
        self.logger.info(f"System prompt updated for Hugging Face model {self.model_name}")