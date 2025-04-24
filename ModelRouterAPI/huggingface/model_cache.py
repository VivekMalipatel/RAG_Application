import logging
import threading
import os
import importlib.util
import shutil
import traceback
import sys
import psutil
from typing import Dict, Tuple, Any, Optional
import time
import torch
from transformers import AutoTokenizer, AutoModel
from model_type import ModelType
from core.device_utils import DeviceManager

logger = logging.getLogger(__name__)

class ModelCache:
    """
    Singleton class for managing and caching Hugging Face models.
    Prevents loading multiple instances of the same model across requests.
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ModelCache, cls).__new__(cls)
                cls._instance._initialize()
            return cls._instance
    
    def _initialize(self):
        """Initialize the model cache."""
        self.models: Dict[str, Any] = {}
        self.tokenizers: Dict[str, Any] = {}
        self.last_used: Dict[str, float] = {}
        self.model_locks: Dict[str, threading.Lock] = {}
        self.max_idle_time = 3600  # 1 hour max idle time before considering removal
        self.cleanup_thread = None
        self.running = True
        self.logger = logging.getLogger(__name__)
        # Get HF token from environment variable
        self.hf_token = os.environ.get("HF_TOKEN")
        
        # Create models directory for our custom HF cache and set environment variables
        # This is the absolute path to the project's model storage
        self.models_dir = os.path.join(os.path.dirname(__file__), ".models")
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Set environment variables to force HF to use our custom cache directory
        os.environ["HF_HOME"] = self.models_dir
        os.environ["TRANSFORMERS_CACHE"] = os.path.join(self.models_dir, "transformers")
        os.environ["HF_DATASETS_CACHE"] = os.path.join(self.models_dir, "datasets")
        os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(self.models_dir, "hub")
        
        self.logger.info(f"Model cache initialized. Models will be stored at: {self.models_dir}")
        
        # Start the cleanup thread
        self._start_cleanup_thread()
    
    def _start_cleanup_thread(self):
        """Start a background thread for cleaning up idle models."""
        def cleanup_worker():
            while self.running:
                try:
                    self._cleanup_idle_models()
                except Exception as e:
                    self.logger.error(f"Error in cleanup thread: {e}")
                time.sleep(300)  # Check every 5 minutes
        
        self.cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        self.cleanup_thread.start()
        self.logger.info("Model cleanup thread started")
    
    def _cleanup_idle_models(self):
        """Remove models that haven't been used for a while to free memory."""
        current_time = time.time()
        models_to_remove = []
        
        # Find idle models
        for model_key, last_used in self.last_used.items():
            if current_time - last_used > self.max_idle_time:
                models_to_remove.append(model_key)
        
        # Remove idle models
        for model_key in models_to_remove:
            try:
                with self.model_locks.get(model_key, threading.Lock()):
                    if model_key in self.models:
                        self.logger.info(f"Removing idle model: {model_key}")
                        del self.models[model_key]
                        if model_key in self.tokenizers:
                            del self.tokenizers[model_key]
                        if model_key in self.last_used:
                            del self.last_used[model_key]
                        # Force GPU memory cleanup if applicable
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
            except Exception as e:
                self.logger.error(f"Error removing model {model_key}: {e}")
    
    def get_model_key(self, model_name: str, model_type: ModelType) -> str:
        """Generate a unique key for the model based on name and type."""
        return f"{model_name}_{model_type.value}"
    
    def _is_nomic_multimodal_model(self, model_name: str) -> bool:
        """Check if the model is a Nomic multimodal model that requires special handling."""
        nomic_models = [
            "nomic-ai/colnomic-embed-multimodal-3b",
            "nomic-ai/colnomic-embed-multimodal-7b",
            "nomic-ai/nomic-embed-multimodal-3b",
            "nomic-ai/nomic-embed-multimodal-7b"
        ]
        return model_name in nomic_models
    
    def _get_colpali_class(self, model_name: str):
        """Get the appropriate colpali model and processor classes based on model name."""
        try:
            # Check if colpali is installed
            if importlib.util.find_spec("colpali_engine") is None:
                self.logger.error("colpali package not found. Please install with: pip install git+https://github.com/illuin-tech/colpali.git")
                raise ImportError("colpali package not found")
            
            from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor, BiQwen2_5, BiQwen2_5_Processor
            
            # Determine the appropriate class based on the model name
            if "colnomic" in model_name:
                return ColQwen2_5, ColQwen2_5_Processor
            else:  # "nomic-embed-multimodal" models
                return BiQwen2_5, BiQwen2_5_Processor
        except ImportError as e:
            self.logger.error(f"Failed to import colpali: {e}")
            raise
    
    def get_model(self, 
                 model_name: str, 
                 model_type: ModelType,
                 device: str,
                 token: Optional[str] = None,
                 trust_remote_code: bool = True) -> Tuple[Any, Any]:
        """
        Get or load a model and its tokenizer.
        
        Args:
            model_name: Name of the Hugging Face model
            model_type: Type of the model (text generation, embedding, reranker)
            device: Device to load the model on ('cuda', 'cpu', 'mps')
            token: Hugging Face API token
            trust_remote_code: Whether to trust remote code in model repos
            
        Returns:
            Tuple of (model, tokenizer)
        """
        model_key = self.get_model_key(model_name, model_type)
        
        # Create a lock for this model if it doesn't exist
        if model_key not in self.model_locks:
            self.model_locks[model_key] = threading.Lock()
        
        # Try to get the model from cache first
        with self.model_locks[model_key]:
            # Update last used timestamp even if just checking
            self.last_used[model_key] = time.time()
            
            # If model is already loaded, return it
            if model_key in self.models and model_key in self.tokenizers:
                self.logger.debug(f"Using cached model: {model_key}")
                return self.models[model_key], self.tokenizers[model_key]
            
            # Otherwise, load the model
            self.logger.info(f"Loading model: {model_name} ({model_type.value}) on {device}")
            
            try:
                # Use token from environment if not provided
                token = token or self.hf_token
                
                # Ensure environment variables are set for this thread
                os.environ["HF_HOME"] = self.models_dir
                os.environ["TRANSFORMERS_CACHE"] = os.path.join(self.models_dir, "transformers")
                os.environ["HF_DATASETS_CACHE"] = os.path.join(self.models_dir, "datasets")
                os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(self.models_dir, "hub")
                
                # Check if this is a Nomic multimodal model that requires special handling
                if self._is_nomic_multimodal_model(model_name):
                    self.logger.info(f"Loading Nomic multimodal model using colpali: {model_name}")
                    try:
                        from transformers.utils.import_utils import is_flash_attn_2_available
                        
                        # Get appropriate colpali model classes
                        ModelClass, ProcessorClass = self._get_colpali_class(model_name)
                        
                        # Load model with colpali
                        model_kwargs = {
                            "torch_dtype": torch.bfloat16,
                            "device_map": device,
                            "cache_dir": os.path.join(self.models_dir, "transformers"),
                            "local_files_only": False
                        }
                        
                        # Only add flash attention if available
                        if is_flash_attn_2_available():
                            model_kwargs["attn_implementation"] = "flash_attention_2"
                            
                        model = ModelClass.from_pretrained(
                            model_name,
                            **model_kwargs
                        ).eval()
                        
                        # Load processor (equivalent to tokenizer for standard models)
                        processor = ProcessorClass.from_pretrained(
                            model_name, 
                            cache_dir=os.path.join(self.models_dir, "transformers")
                        )
                        
                        # Cache the model and processor
                        self.models[model_key] = model
                        self.tokenizers[model_key] = processor
                        self.last_used[model_key] = time.time()
                        
                        self.logger.info(f"Successfully loaded Nomic model with colpali: {model_name}")
                        return model, processor
                    
                    except Exception as e:
                        self.logger.error(f"Failed to load Nomic model with colpali: {e}")
                        self.logger.error(f"Error details: {traceback.format_exc()}")
                        raise
                else:
                    # Standard HuggingFace model loading
                    # Load tokenizer
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_name, 
                        token=token,
                        cache_dir=os.path.join(self.models_dir, "transformers")
                    )
                    
                    # Load model
                    model = AutoModel.from_pretrained(
                        model_name, 
                        trust_remote_code=trust_remote_code, 
                        revision="main", 
                        token=token,
                        device_map=device,
                        cache_dir=os.path.join(self.models_dir, "transformers")
                    )
                    
                    # Set to evaluation mode
                    model.eval()
                    model.to(device)
                    
                    # Store in cache
                    self.models[model_key] = model
                    self.tokenizers[model_key] = tokenizer
                    self.last_used[model_key] = time.time()
                    
                    self.logger.info(f"Successfully loaded model: {model_name}")
                    
                    return model, tokenizer
                
            except Exception as e:
                self.logger.error(f"Error loading model {model_name}: {e}")
                self.logger.error(f"Error details: {traceback.format_exc()}")
                raise
    
    def shutdown(self):
        """Clean up resources and stop background threads."""
        self.running = False
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=1.0)
        
        # Clear all models
        model_keys = list(self.models.keys())
        for model_key in model_keys:
            try:
                with self.model_locks.get(model_key, threading.Lock()):
                    if model_key in self.models:
                        del self.models[model_key]
                        if model_key in self.tokenizers:
                            del self.tokenizers[model_key]
            except Exception as e:
                self.logger.error(f"Error during shutdown, clearing model {model_key}: {e}")
        
        # Clear GPU cache if applicable
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.logger.info("Model cache shutdown complete")