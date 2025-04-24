import logging
import threading
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
        self.models: Dict[str, Dict[str, Any]] = {}
        self.tokenizers: Dict[str, Any] = {}
        self.last_used: Dict[str, float] = {}
        self.model_locks: Dict[str, threading.Lock] = {}
        self.max_idle_time = 3600  # 1 hour max idle time before considering removal
        self.cleanup_thread = None
        self.running = True
        self.logger = logging.getLogger(__name__)
        self.logger.info("Model cache initialized")
        
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
                        # Force GPU/MPS memory cleanup if applicable
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        elif DeviceManager.is_mps_available():
                            torch.mps.empty_cache()
            except Exception as e:
                self.logger.error(f"Error removing model {model_key}: {e}")
    
    def get_model_key(self, model_name: str, model_type: ModelType) -> str:
        """Generate a unique key for the model based on name and type."""
        return f"{model_name}_{model_type.value}"
    
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
                # Get device-specific loading parameters
                load_kwargs = DeviceManager.get_model_kwargs(device)
                
                # Add common kwargs
                load_kwargs.update({
                    "trust_remote_code": trust_remote_code,
                    "revision": "main",
                    "token": token
                })
                
                # Load tokenizer
                tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
                
                # Load model
                try:
                    model = AutoModel.from_pretrained(model_name, **load_kwargs)
                    model.eval()  # Set to evaluation mode
                except Exception as e:
                    # If there's an error with specific device configurations, try simpler approach
                    self.logger.warning(f"Error with device-specific loading, trying fallback: {str(e)}")
                    
                    # Remove device map settings that might cause issues
                    if "device_map" in load_kwargs:
                        del load_kwargs["device_map"]
                        
                    model = AutoModel.from_pretrained(model_name, **load_kwargs)
                    model.eval()
                    
                    # Move model to device after loading if needed
                    if device != "cpu":
                        try:
                            model = model.to(device)
                        except Exception as device_err:
                            self.logger.warning(f"Failed to move model to {device}, falling back to CPU: {str(device_err)}")
                            model = model.to("cpu")
                
                # Store in cache
                self.models[model_key] = model
                self.tokenizers[model_key] = tokenizer
                self.last_used[model_key] = time.time()
                
                self.logger.info(f"Successfully loaded model: {model_name} on {model.device if hasattr(model, 'device') else device}")
                
                return model, tokenizer
                
            except Exception as e:
                self.logger.error(f"Error loading model {model_name}: {e}")
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
        
        # Clear GPU/MPS cache if applicable
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif DeviceManager.is_mps_available():
            torch.mps.empty_cache()
        
        self.logger.info("Model cache shutdown complete")