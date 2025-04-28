import torch
import logging
import asyncio
import gc
from transformers import AutoTokenizer, AutoModel, TextIteratorStreamer
from typing import List, Union, Optional, AsyncGenerator, Dict
import threading
import requests
import numpy as np
from config import settings
from model_type import ModelType
from huggingface.model_cache import ModelCache
from core.device_utils import DeviceManager

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
        self.logger = logging.getLogger(__name__)
        
        # Check available devices and determine optimal device based on memory
        self._determine_optimal_device(device)
        
        self.model_name = model_name
        self.model_type = model_type
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.stream = stream
        self.trust_remote_code = trust_remote_code
        self.hf_token = settings.HUGGINGFACE_API_TOKEN
        
        # Store the model key for later use with the cache
        self.model_key = None

        self.model = None
        self.tokenizer = None
        self.processor = None
        self.cpu_model = None  # Reference to CPU model for fallback
        
        # Estimate memory requirements per sample for different model types
        # These are rough estimates and may need tuning
        self.memory_requirements = {
            ModelType.TEXT_GENERATION: 0.5,  # GB per sample
            ModelType.TEXT_EMBEDDING: 0.1,   # GB per sample
            ModelType.IMAGE_EMBEDDING: 0.3,  # GB per sample
            ModelType.RERANKER: 0.2,         # GB per sample
        }

        if not self.is_model_available():
            raise ValueError(f"Model {self.model_name} is not available on Hugging Face.")

        try:
            self._load_model()
        except Exception as e:
            self.logger.error(f"Failed to load Hugging Face model {model_name}: {str(e)}")
            raise RuntimeError(f"Model initialization failed: {model_name}")
    
    def _determine_optimal_device(self, requested_device: Optional[str] = None):
        """Determine the optimal device based on available hardware and memory."""
        # First, respect explicitly requested device if provided
        if requested_device in ["cuda", "cpu", "mps"]:
            if requested_device == "cuda" and not torch.cuda.is_available():
                self.logger.warning("CUDA explicitly requested but not available, falling back to CPU.")
                self.device = "cpu"
            elif requested_device == "mps" and not torch.backends.mps.is_available():
                self.logger.warning("MPS explicitly requested but not available, falling back to CPU.")
                self.device = "cpu"
            else:
                self.device = requested_device
            return
            
        # If no specific device requested, use DeviceManager to find optimal device
        self.device = DeviceManager.get_optimal_device()
        
        # If optimal device is CUDA, check if it has sufficient memory
        if self.device == "cuda":
            cuda_available, cuda_message = DeviceManager.check_cuda_availability_with_memory_threshold()
            self.logger.info(f"CUDA availability check: {cuda_message}")
            if not cuda_available:
                self.logger.warning("CUDA available but has insufficient memory, falling back to CPU.")
                self.device = "cpu"
    
    def _get_model_specific_memory_estimate(self) -> float:
        """Get a model-specific memory estimate in GB."""
        model_size_map = {
            "3b": 3.0,
            "4b": 4.0,
            "7b": 7.0,
            "8b": 8.0,
            "14b": 14.0,
            "large": 1.5,
            "small": 0.5,
            "base": 0.8,
            "tiny": 0.2,
            "mini": 0.3,
        }
        
        # Default memory estimate
        memory_estimate = 1.0  # 1 GB
        
        # Check if model name contains any size indicators
        for size_key, size_value in model_size_map.items():
            if size_key in self.model_name.lower():
                memory_estimate = size_value
                break
        
        return memory_estimate
    
    def _load_model(self):
        """Load the model with memory-aware approach."""
        model_cache = ModelCache()
        model_memory_estimate = self._get_model_specific_memory_estimate()
        self.logger.info(f"Estimated memory requirement for {self.model_name}: {model_memory_estimate}GB")
        
        is_nomic_multimodal = any(model_id in self.model_name for model_id in [
            "nomic-ai/colnomic-embed-multimodal",
            "nomic-ai/nomic-embed-multimodal"
        ])
        
        if is_nomic_multimodal:
            self.logger.info(f"Loading Nomic multimodal model: {self.model_name}")
        
        try:
            # Use model cache to get model - it implements the sophisticated device management
            self.model, self.tokenizer = model_cache.get_model(
                model_name=self.model_name,
                model_type=self.model_type,
                device=self.device,
                token=self.hf_token,
                trust_remote_code=self.trust_remote_code
            )
            
            # Store the model key for later use with cache's get_model_for_inference
            self.model_key = model_cache.get_model_key(self.model_name, self.model_type)
            
            if is_nomic_multimodal:
                self.processor = self.tokenizer
                self.logger.info(f"Stored processor for Nomic multimodal model: {self.model_name}")
                
            self.logger.info(f"Model loaded successfully: {self.model_name} on {self.device}")
        except Exception as e:
            self.logger.error(f"Model loading failed: {str(e)}")
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

    # async def generate_text(self, prompt: str, max_tokens: Optional[int] = None, stream: Optional[bool] = None) -> Union[str, AsyncGenerator[str, None]]:
    #     full_prompt = prompt
    #     if self.system_prompt:
    #         full_prompt = f"{self.system_prompt}\n\n{prompt}"
        
    #     inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
    #     stream = stream if stream is not None else self.stream
        
    #     if self.stream:
    #         streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)
    #         generation_kwargs = {
    #             **inputs,
    #             "max_length": max_tokens if max_tokens else self.max_tokens,
    #             "temperature": self.temperature,
    #             "top_p": self.top_p,
    #             "streamer": streamer
    #         }
            
    #         threading.Thread(target=self.model.generate, kwargs=generation_kwargs).start()
            
    #         async def stream_generator():
    #             for text in streamer:
    #                 yield text
    #         return stream_generator()
    #     else:
    #         output = self.model.generate(
    #             **inputs,
    #             max_length=max_tokens if max_tokens else self.max_tokens,
    #             temperature=self.temperature,
    #             top_p=self.top_p,
    #         )
    #         return self.tokenizer.decode(output[0], skip_special_tokens=True)
    
    async def embed_text(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts with dynamic device selection."""
        try:
            batch_size = len(texts)
            self._prepare_for_inference(batch_size, ModelType.TEXT_EMBEDDING)
            
            # Use the proper approach based on model type
            if "colnomic-embed-multimodal" in self.model_name or "nomic-embed-multimodal" in self.model_name:
                return await self._embed_text_nomic(texts)
            else:
                return await self._embed_text_standard(texts)
                
        except torch.cuda.OutOfMemoryError as e:
            self.logger.warning(f"GPU out of memory during text embedding: {e}")
            # Try to recover by switching to CPU and clearing GPU memory
            return await self._handle_oom_error(lambda: self._embed_text_with_device(texts, "cpu"))
            
        except Exception as e:
            self.logger.error(f"Error in embed_text: {str(e)}")
            raise
    
    async def _embed_text_nomic(self, texts: List[str]) -> List[List[float]]:
        """Handle embedding specifically for Nomic multimodal models."""
        self.logger.info(f"Using specialized processing for Nomic multimodal model: {self.model_name}")
        try:
            max_wait_time = 30
            wait_time = 0
            check_interval = 0.5 

            while self.processor is None and wait_time < max_wait_time:
                self.logger.warning(f"Processor not yet initialized, waiting... ({wait_time}s/{max_wait_time}s)")
                await asyncio.sleep(check_interval)
                wait_time += check_interval
            
            # Get model for inference with appropriate device
            model_cache = ModelCache()
            inference_model, actual_device = model_cache.get_model_for_inference(self.model_key, len(texts))
            
            batch_queries = self.processor.process_queries(texts)
            batch_queries = batch_queries.to(device=actual_device, dtype=torch.float32 if actual_device == 'cpu' else None)

            if actual_device == 'cpu' and hasattr(inference_model, 'to'):
                inference_model = inference_model.to(dtype=torch.float32)
            
            with torch.no_grad():
                query_embeddings = inference_model(**batch_queries)
            
            if query_embeddings.dtype != torch.float32:
                query_embeddings = query_embeddings.to(torch.float32)
            
            # Release memory if possible
            if actual_device == "cuda":
                batch_queries = batch_queries.cpu()
                torch.cuda.empty_cache()
            
            return query_embeddings.cpu().numpy().tolist()
        except Exception as e:
            self.logger.error(f"Error generating embeddings with Nomic model: {e}")
            raise
    
    async def _embed_text_standard(self, texts: List[str]) -> List[List[float]]:
        """Handle embedding for standard models."""
        # Get model for inference with appropriate device
        model_cache = ModelCache()
        inference_model, actual_device = model_cache.get_model_for_inference(self.model_key, len(texts))
        
        # Process in smaller batches if the batch is large
        batch_size = len(texts)
        if batch_size > 32 and actual_device == "cuda":
            return await self._process_in_batches(texts, 32, self._embed_text_batch)
        
        return await self._embed_text_batch(texts, inference_model, actual_device)
    
    async def _embed_text_batch(self, texts: List[str], model=None, device=None) -> List[List[float]]:
        """Process a single batch of text embeddings."""
        if model is None or device is None:
            model_cache = ModelCache()
            model, device = model_cache.get_model_for_inference(self.model_key, len(texts))
            
        inputs = self.tokenizer(texts, padding=True, return_tensors="pt").to(device)
        
        with torch.no_grad():
            try:
                embeddings = model(**inputs).last_hidden_state.mean(dim=1)
                
                if embeddings.dtype != torch.float32:
                    embeddings = embeddings.to(torch.float32)
                
                # Release memory if possible
                if device == "cuda":
                    inputs = {k: v.cpu() for k, v in inputs.items()}
                    torch.cuda.empty_cache()
                
                return embeddings.cpu().numpy().tolist()
            except Exception as e:
                self.logger.error(f"Error in _embed_text_batch: {e}")
                raise
    
    async def _process_in_batches(self, items, batch_size, batch_func):
        """Process items in smaller batches."""
        results = []
        for i in range(0, len(items), batch_size):
            batch_items = items[i:i+batch_size]
            batch_results = await batch_func(batch_items)
            results.extend(batch_results)
            # Clean up between batches if using GPU
            if self.device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
        return results
        
    async def _handle_oom_error(self, recovery_func):
        """Handle Out-of-Memory errors by clearing GPU memory and falling back to CPU."""
        self.logger.warning("Handling OutOfMemory error by clearing GPU memory and retrying with CPU")
        DeviceManager.clear_gpu_memory()
        
        try:
            # Try recovery function (which should use CPU)
            return await recovery_func()
        except Exception as e:
            self.logger.error(f"Recovery function also failed after OOM: {e}")
            raise
    
    async def _embed_text_with_device(self, texts: List[str], device: str) -> List[List[float]]:
        """Force embedding on a specific device."""
        self.logger.info(f"Forcing text embedding on device: {device}")
        
        # Get model for the specific device
        model_cache = ModelCache()
        model_key = model_cache.get_model_key(self.model_name, self.model_type)
        
        # Use model_cache to get the right model for the device
        if device == "cpu" and model_key in model_cache.cpu_models:
            model = model_cache.cpu_models[model_key]
        elif device == "cuda" and model_key in model_cache.gpu_models:
            model = model_cache.gpu_models[model_key]
        else:
            self.logger.error(f"No model available for device {device}, cannot recover from OOM")
            raise RuntimeError(f"No model available for device {device}")
        
        if "colnomic-embed-multimodal" in self.model_name or "nomic-embed-multimodal" in self.model_name:
            # Handle Nomic models
            batch_queries = self.processor.process_queries(texts)
            batch_queries = batch_queries.to(device=device, dtype=torch.float32 if device == 'cpu' else None)
            
            if device == 'cpu' and hasattr(model, 'to'):
                model = model.to(dtype=torch.float32)
            
            with torch.no_grad():
                query_embeddings = model(**batch_queries)
            
            if query_embeddings.dtype != torch.float32:
                query_embeddings = query_embeddings.to(torch.float32)
            
            return query_embeddings.cpu().numpy().tolist()
        else:
            # Handle standard models
            inputs = self.tokenizer(texts, padding=True, return_tensors="pt").to(device)
            
            with torch.no_grad():
                embeddings = model(**inputs).last_hidden_state.mean(dim=1)
                
                if embeddings.dtype != torch.float32:
                    embeddings = embeddings.to(torch.float32)
                
                return embeddings.cpu().numpy().tolist()
    
    def _prepare_for_inference(self, batch_size: int, model_type: ModelType = None):
        """Prepare for inference by checking if device is suitable for the batch size."""
        if model_type is None:
            model_type = self.model_type
            
        # Check if memory is sufficient for this operation on current device
        if self.device == "cuda":
            # Estimate memory needed for this batch
            memory_per_item = self.memory_requirements.get(model_type, 0.1)  # Default to 0.1 GB if unknown
            estimated_memory = batch_size * memory_per_item
            
            # Log memory usage before inference
            stats = DeviceManager.get_memory_stats()
            if "gpu_free" in stats:
                self.logger.info(f"GPU free memory before inference: {stats['gpu_free']:.2f}GB, estimated need: {estimated_memory:.2f}GB")
                
            # Check if GPU can handle this batch
            if not DeviceManager.is_gpu_suitable_for_inference(batch_size, memory_per_item):
                self.logger.warning(
                    f"GPU may not have enough memory for batch size {batch_size} (est. {estimated_memory:.2f}GB required). "
                    f"Will attempt inference but may fall back to CPU if OOM occurs."
                )
                
    async def embed_image(self, images: List[dict]) -> List[List[float]]:
        """Generate embeddings for a list of images with dynamic device selection."""
        try:
            batch_size = len(images)
            self._prepare_for_inference(batch_size, ModelType.IMAGE_EMBEDDING)
            
            if "colnomic-embed-multimodal" in self.model_name or "nomic-embed-multimodal" in self.model_name:
                return await self._embed_image_nomic(images)
            else:
                self.logger.error("Image embedding is not supported for this model.")
                raise NotImplementedError("Image embedding is not supported for this model.")
                
        except torch.cuda.OutOfMemoryError as e:
            self.logger.warning(f"GPU out of memory during image embedding: {e}")
            # Try to recover by switching to CPU and clearing GPU memory
            return await self._handle_oom_error(lambda: self._embed_image_with_device(images, "cpu"))
            
        except Exception as e:
            self.logger.error(f"Error in embed_image: {str(e)}")
            raise
    
    async def _embed_image_nomic(self, images: List[dict]) -> List[List[float]]:
        """Handle image embedding for Nomic models."""
        self.logger.info(f"Using specialized processing for Nomic multimodal model: {self.model_name}")
        try:
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
            
            # Get model for inference with appropriate device
            model_cache = ModelCache()
            inference_model, actual_device = model_cache.get_model_for_inference(self.model_key, len(images))
            
            processed_images = []
            context_prompts = []
            
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
            
            # Process in smaller batches if the batch size is large
            if len(processed_images) > 8 and actual_device == "cuda":
                return await self._process_image_batches(processed_images, context_prompts, 8, actual_device, inference_model)
            
            batch_images = self.processor.process_images(
                images=processed_images, 
                context_prompts=context_prompts
            )
            
            batch_images = batch_images.to(device=actual_device, dtype=torch.float32 if actual_device == 'cpu' else None)

            if actual_device == 'cpu' and hasattr(inference_model, 'to'):
                inference_model = inference_model.to(dtype=torch.float32)
            
            with torch.no_grad():
                image_embeddings = inference_model(**batch_images)
            
            if image_embeddings.dtype != torch.float32:
                image_embeddings = image_embeddings.to(torch.float32)
            
            # Release memory if possible
            if actual_device == "cuda":
                batch_images = batch_images.cpu()
                torch.cuda.empty_cache()
            
            return image_embeddings.cpu().numpy().tolist()
        except Exception as e:
            self.logger.error(f"Error generating embeddings with Nomic model: {e}")
            raise
    
    async def _process_image_batches(self, images, prompts, batch_size, device, model):
        """Process image embeddings in smaller batches."""
        results = []
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            batch_prompts = prompts[i:i+batch_size]
            
            batch_processed = self.processor.process_images(
                images=batch_images, 
                context_prompts=batch_prompts
            )
            
            batch_processed = batch_processed.to(device=device, dtype=torch.float32 if device == 'cpu' else None)
            
            with torch.no_grad():
                batch_embeddings = model(**batch_processed)
                
            if batch_embeddings.dtype != torch.float32:
                batch_embeddings = batch_embeddings.to(torch.float32)
                
            batch_results = batch_embeddings.cpu().numpy().tolist()
            results.extend(batch_results)
            
            # Clean up after batch
            if device == "cuda":
                del batch_processed
                torch.cuda.empty_cache()
                gc.collect()
                
        return results
    
    async def _embed_image_with_device(self, images: List[dict], device: str) -> List[List[float]]:
        """Force image embedding on a specific device."""
        if not ("colnomic-embed-multimodal" in self.model_name or "nomic-embed-multimodal" in self.model_name):
            raise NotImplementedError("Image embedding is only supported for Nomic models")
            
        self.logger.info(f"Forcing image embedding on device: {device}")
        
        # Get model for the specific device
        model_cache = ModelCache()
        model_key = model_cache.get_model_key(self.model_name, self.model_type)
        
        # Use model_cache to get the right model for the device
        if device == "cpu" and model_key in model_cache.cpu_models:
            model = model_cache.cpu_models[model_key]
        elif device == "cuda" and model_key in model_cache.gpu_models:
            model = model_cache.gpu_models[model_key]
        else:
            self.logger.error(f"No model available for device {device}, cannot recover from OOM")
            raise RuntimeError(f"No model available for device {device}")
        
        import base64
        from io import BytesIO
        from PIL import Image
        
        processed_images = []
        context_prompts = []
        
        for image_data in images:
            base64_str = image_data["image"]
            img_data = base64.b64decode(base64_str)
            img_bytes = BytesIO(img_data)
            img = Image.open(img_bytes)
            processed_images.append(img)
            context_prompts.append(image_data["text"])
        
        # Process in smaller batches for CPU
        if len(processed_images) > 4 and device == "cpu":
            return await self._process_image_batches(processed_images, context_prompts, 4, device, model)
        
        batch_images = self.processor.process_images(
            images=processed_images, 
            context_prompts=context_prompts
        )
        
        batch_images = batch_images.to(device=device, dtype=torch.float32 if device == 'cpu' else None)

        if device == 'cpu' and hasattr(model, 'to'):
            model = model.to(dtype=torch.float32)
        
        with torch.no_grad():
            image_embeddings = model(**batch_images)
        
        if image_embeddings.dtype != torch.float32:
            image_embeddings = image_embeddings.to(torch.float32)
        
        return image_embeddings.cpu().numpy().tolist()

    async def rerank_documents(self, query: str, documents: List[str], max_tokens: int) -> List[int]:
        """Rerank documents with dynamic device selection and memory management."""
        try:
            batch_size = len(documents)
            self._prepare_for_inference(batch_size, ModelType.RERANKER)
            
            # Get model for inference with appropriate device
            model_cache = ModelCache()
            inference_model, actual_device = model_cache.get_model_for_inference(self.model_key, batch_size)
            
            query_tokens = self.tokenizer(query, return_tensors="pt").to(actual_device)
            
            # Check for very long documents that might cause memory issues
            if len(documents) * max_tokens > 8000:
                self.logger.warning(f"Large batch of {len(documents)} documents with max_tokens={max_tokens}, may exceed memory limits")
                truncated_documents = []
                for doc in documents:
                    truncated_documents.append(doc[:max_tokens-5]+".....")
                documents = truncated_documents
            
            doc_tokens = self.tokenizer(documents, return_tensors="pt", padding=True).to(actual_device)

            with torch.no_grad():
                query_embedding = inference_model(**query_tokens).last_hidden_state.mean(dim=1)
                doc_embeddings = inference_model(**doc_tokens).last_hidden_state.mean(dim=1)
                scores = torch.matmul(query_embedding, doc_embeddings.T).squeeze().cpu().numpy()

            # Clear memory if using GPU
            if actual_device == "cuda":
                query_tokens = {k: v.cpu() for k, v in query_tokens.items()}
                doc_tokens = {k: v.cpu() for k, v in doc_tokens.items()}
                torch.cuda.empty_cache()

            return np.argsort(scores)[::-1].tolist()
            
        except torch.cuda.OutOfMemoryError as e:
            self.logger.warning(f"GPU out of memory during reranking: {e}")
            DeviceManager.clear_gpu_memory()
            
            # Fall back to CPU
            return await self._rerank_documents_with_device(query, documents, max_tokens, "cpu")
            
        except Exception as e:
            self.logger.error(f"Error in rerank_documents: {str(e)}")
            raise
    
    async def _rerank_documents_with_device(self, query: str, documents: List[str], max_tokens: int, device: str) -> List[int]:
        """Force reranking on a specific device."""
        self.logger.info(f"Forcing document reranking on device: {device}")
        
        # Get model for the specific device
        model_cache = ModelCache()
        model_key = model_cache.get_model_key(self.model_name, self.model_type)
        
        # Use model_cache to get the right model for the device
        if device == "cpu" and model_key in model_cache.cpu_models:
            model = model_cache.cpu_models[model_key]
        elif device == "cuda" and model_key in model_cache.gpu_models:
            model = model_cache.gpu_models[model_key]
        else:
            self.logger.error(f"No model available for device {device}, cannot recover from OOM")
            raise RuntimeError(f"No model available for device {device}")
            
        # Check for very long documents
        if len(documents) * max_tokens > 8000:
            truncated_documents = []
            for doc in documents:
                truncated_documents.append(doc[:max_tokens-5]+".....")
            documents = truncated_documents
            
        query_tokens = self.tokenizer(query, return_tensors="pt").to(device)
        doc_tokens = self.tokenizer(documents, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            query_embedding = model(**query_tokens).last_hidden_state.mean(dim=1)
            doc_embeddings = model(**doc_tokens).last_hidden_state.mean(dim=1)
            scores = torch.matmul(query_embedding, doc_embeddings.T).squeeze().cpu().numpy()

        return np.argsort(scores)[::-1].tolist()

    def set_system_prompt(self, system_prompt: str):
        self.system_prompt = system_prompt
        self.logger.info(f"System prompt updated for Hugging Face model {self.model_name}")
        
    def __del__(self):
        """Clean up resources when the client is destroyed."""
        try:
            if torch.cuda.is_available():
                DeviceManager.clear_gpu_memory()
        except:
            pass