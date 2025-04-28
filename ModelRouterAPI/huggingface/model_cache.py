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
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ModelCache, cls).__new__(cls)
                cls._instance._initialize()
            return cls._instance
    
    def _initialize(self):
        self.models: Dict[str, Any] = {}
        self.tokenizers: Dict[str, Any] = {}
        self.last_used: Dict[str, float] = {}
        self.model_locks: Dict[str, threading.Lock] = {}
        self.max_idle_time = 3600
        self.cleanup_thread = None
        self.running = True
        self.logger = logging.getLogger(__name__)
        self.hf_token = os.environ.get("HF_TOKEN")
        # Keep track of both GPU and CPU versions of models
        self.gpu_models: Dict[str, Any] = {}
        self.cpu_models: Dict[str, Any] = {}
        self.model_placement: Dict[str, str] = {}  # Stores which device the model is currently using
        
        self.models_dir = os.path.join(os.path.dirname(__file__), ".models")
        os.makedirs(self.models_dir, exist_ok=True)
        
        os.environ["HF_HOME"] = self.models_dir
        os.environ["TRANSFORMERS_CACHE"] = os.path.join(self.models_dir, "transformers")
        os.environ["HF_DATASETS_CACHE"] = os.path.join(self.models_dir, "datasets")
        os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(self.models_dir, "hub")
        
        self.logger.info(f"Model cache initialized. Models will be stored at: {self.models_dir}")
        
        self._start_cleanup_thread()
    
    def _start_cleanup_thread(self):
        def cleanup_worker():
            while self.running:
                try:
                    self._cleanup_idle_models()
                except Exception as e:
                    self.logger.error(f"Error in cleanup thread: {e}")
                time.sleep(300)
        
        self.cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        self.cleanup_thread.start()
        self.logger.info("Model cleanup thread started")
    
    def _cleanup_idle_models(self):
        current_time = time.time()
        models_to_remove = []
        
        for model_key, last_used in self.last_used.items():
            if current_time - last_used > self.max_idle_time:
                models_to_remove.append(model_key)
        
        for model_key in models_to_remove:
            try:
                with self.model_locks.get(model_key, threading.Lock()):
                    # Clean up both GPU and CPU versions if they exist
                    if model_key in self.gpu_models:
                        self.logger.info(f"Removing idle GPU model: {model_key}")
                        del self.gpu_models[model_key]
                        
                    if model_key in self.cpu_models:
                        self.logger.info(f"Removing idle CPU model: {model_key}")
                        del self.cpu_models[model_key]
                        
                    if model_key in self.models:
                        del self.models[model_key]
                        
                    if model_key in self.tokenizers:
                        del self.tokenizers[model_key]
                        
                    if model_key in self.last_used:
                        del self.last_used[model_key]
                        
                    if model_key in self.model_placement:
                        del self.model_placement[model_key]
                        
                    # Clean GPU memory if possible
                    if torch.cuda.is_available():
                        DeviceManager.clear_gpu_memory()
            except Exception as e:
                self.logger.error(f"Error removing model {model_key}: {e}")
    
    def get_model_key(self, model_name: str, model_type: ModelType) -> str:
        return f"{model_name}"
    
    def _is_nomic_multimodal_model(self, model_name: str) -> bool:
        nomic_models = [
            "nomic-ai/colnomic-embed-multimodal-3b",
            "nomic-ai/colnomic-embed-multimodal-7b",
            "nomic-ai/nomic-embed-multimodal-3b",
            "nomic-ai/nomic-embed-multimodal-7b"
        ]
        return model_name in nomic_models
    
    def _get_colpali_class(self, model_name: str):
        try:
            if importlib.util.find_spec("colpali_engine") is None:
                self.logger.error("colpali package not found. Please install with: pip install git+https://github.com/illuin-tech/colpali.git")
                raise ImportError("colpali package not found")
            
            from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor, BiQwen2_5, BiQwen2_5_Processor
            
            if "colnomic" in model_name:
                return ColQwen2_5, ColQwen2_5_Processor
            else:
                return BiQwen2_5, BiQwen2_5_Processor
        except ImportError as e:
            self.logger.error(f"Failed to import colpali: {e}")
            raise
    
    def _load_model_with_device_fallback(self, model_name, model_type, device, token=None, trust_remote_code=True):
        """Load a model with device fallback if the primary device doesn't have enough memory."""
        model_key = self.get_model_key(model_name, model_type)
        
        # First, attempt to determine if the requested device can handle the model
        if device == "cuda":
            cuda_available, cuda_message = DeviceManager.check_cuda_availability_with_memory_threshold()
            self.logger.info(f"CUDA availability check: {cuda_message}")
            
            if cuda_available:
                try:
                    # Try loading on CUDA first
                    self.logger.info(f"Loading model {model_name} on CUDA")
                    model, tokenizer = self._load_model_on_specific_device(
                        model_name, model_type, "cuda", token, trust_remote_code
                    )
                    self.gpu_models[model_key] = model
                    self.model_placement[model_key] = "cuda"
                    
                    # Preemptively load CPU version for fallback during high GPU usage
                    cpu_available, cpu_message = DeviceManager.check_cpu_memory_availability()
                    if cpu_available:
                        # Load model on CPU in background to prepare for potential fallback
                        thread = threading.Thread(
                            target=self._load_cpu_model_background,
                            args=(model_name, model_type, token, trust_remote_code, model_key)
                        )
                        thread.daemon = True
                        thread.start()
                    else:
                        self.logger.warning(f"Can't preload CPU model for fallback: {cpu_message}")
                        
                    return model, tokenizer
                except torch.cuda.OutOfMemoryError as e:
                    self.logger.warning(f"CUDA out of memory when loading model {model_name}: {e}")
                    DeviceManager.clear_gpu_memory()
                    # Fall through to CPU loading
                except Exception as e:
                    self.logger.error(f"Error loading model on CUDA: {e}")
                    self.logger.error(traceback.format_exc())
                    DeviceManager.clear_gpu_memory()
                    # Fall through to CPU loading
            
        # If CUDA failed or wasn't available, try CPU
        cpu_available, cpu_message = DeviceManager.check_cpu_memory_availability()
        self.logger.info(f"CPU availability check: {cpu_message}")
        
        if cpu_available:
            try:
                self.logger.info(f"Loading model {model_name} on CPU")
                model, tokenizer = self._load_model_on_specific_device(
                    model_name, model_type, "cpu", token, trust_remote_code
                )
                self.cpu_models[model_key] = model
                self.model_placement[model_key] = "cpu"
                return model, tokenizer
            except Exception as e:
                self.logger.error(f"Error loading model on CPU: {e}")
                self.logger.error(traceback.format_exc())
                raise RuntimeError(f"Failed to load model {model_name} on any device")
        else:
            raise RuntimeError(f"Insufficient memory on both GPU and CPU to load model {model_name}")
    
    def _load_cpu_model_background(self, model_name, model_type, token, trust_remote_code, model_key):
        """Load CPU version of model in background for potential fallback."""
        try:
            self.logger.info(f"Preloading CPU version of model {model_name} in background")
            model, _ = self._load_model_on_specific_device(model_name, model_type, "cpu", token, trust_remote_code)
            self.cpu_models[model_key] = model
            self.logger.info(f"Successfully preloaded CPU version of model {model_name}")
        except Exception as e:
            self.logger.error(f"Failed to preload CPU model {model_name} in background: {e}")
    
    def _load_model_on_specific_device(self, model_name, model_type, device, token, trust_remote_code):
        """Load model on a specific device."""
        if self._is_nomic_multimodal_model(model_name):
            return self._load_nomic_model(model_name, device, token, trust_remote_code)
        else:
            return self._load_standard_model(model_name, device, token, trust_remote_code)
            
    def _load_nomic_model(self, model_name, device, token, trust_remote_code):
        """Load a Nomic multimodal model."""
        self.logger.info(f"Loading Nomic multimodal model using colpali: {model_name} on {device}")
        try:
            from transformers.utils.import_utils import is_flash_attn_2_available
            
            ModelClass, ProcessorClass = self._get_colpali_class(model_name)
            
            model_kwargs = {
                "torch_dtype": torch.bfloat16 if device != "cpu" else torch.float32,
                "device_map": device,
                "cache_dir": os.path.join(self.models_dir, "transformers"),
                "local_files_only": False
            }
            
            if device != "cpu" and is_flash_attn_2_available():
                model_kwargs["attn_implementation"] = "flash_attention_2"
                
            model = ModelClass.from_pretrained(
                model_name,
                **model_kwargs
            ).eval()
            
            processor = ProcessorClass.from_pretrained(
                model_name, 
                cache_dir=os.path.join(self.models_dir, "transformers")
            )
            
            self.logger.info(f"Successfully loaded Nomic model with colpali: {model_name} on {device}")
            return model, processor
        
        except Exception as e:
            self.logger.error(f"Failed to load Nomic model with colpali on {device}: {e}")
            self.logger.error(f"Error details: {traceback.format_exc()}")
            raise
            
    def _load_standard_model(self, model_name, device, token, trust_remote_code):
        """Load a standard HuggingFace model."""
        self.logger.info(f"Loading standard model: {model_name} on {device}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                token=token,
                cache_dir=os.path.join(self.models_dir, "transformers")
            )
            
            model = AutoModel.from_pretrained(
                model_name, 
                trust_remote_code=trust_remote_code, 
                revision="main", 
                token=token,
                device_map=device,
                torch_dtype=torch.float16 if device != "cpu" else torch.float32,
                cache_dir=os.path.join(self.models_dir, "transformers")
            )
            
            model.eval()
            
            self.logger.info(f"Successfully loaded model: {model_name} on {device}")
            return model, tokenizer
        except Exception as e:
            self.logger.error(f"Error loading model {model_name} on {device}: {e}")
            self.logger.error(f"Error details: {traceback.format_exc()}")
            raise
            
    def get_model(self, 
                 model_name: str, 
                 model_type: ModelType,
                 device: str,
                 token: Optional[str] = None,
                 trust_remote_code: bool = True) -> Tuple[Any, Any]:
        model_key = self.get_model_key(model_name, model_type)
        
        if model_key not in self.model_locks:
            self.model_locks[model_key] = threading.Lock()
        
        with self.model_locks[model_key]:
            self.last_used[model_key] = time.time()
            
            # Check if model is already loaded on either device
            if model_key in self.model_placement:
                current_device = self.model_placement[model_key]
                self.logger.debug(f"Model {model_key} is currently on {current_device}, requested device is {device}")
                
                # If model is already on the requested device, return it
                if current_device == device and ((device == "cuda" and model_key in self.gpu_models) or 
                                               (device == "cpu" and model_key in self.cpu_models)):
                    if device == "cuda":
                        self.logger.debug(f"Using cached GPU model: {model_key}")
                        self.models[model_key] = self.gpu_models[model_key]
                    else:
                        self.logger.debug(f"Using cached CPU model: {model_key}")
                        self.models[model_key] = self.cpu_models[model_key]
                    return self.models[model_key], self.tokenizers[model_key]
                
                # Check if we need to fallback to CPU (if requested CUDA but only have CPU version)
                if device == "cuda" and current_device == "cpu" and model_key in self.cpu_models:
                    # Try to load on GPU now if memory permits
                    cuda_available, _ = DeviceManager.check_cuda_availability_with_memory_threshold()
                    if cuda_available:
                        try:
                            self.logger.info(f"Attempting to move model {model_key} from CPU to GPU")
                            model, tokenizer = self._load_model_on_specific_device(
                                model_name, model_type, "cuda", token, trust_remote_code
                            )
                            self.gpu_models[model_key] = model
                            self.model_placement[model_key] = "cuda"
                            self.models[model_key] = model
                            return model, tokenizer
                        except Exception as e:
                            self.logger.warning(f"Failed to move model to GPU, will use CPU version: {e}")
                            self.models[model_key] = self.cpu_models[model_key]
                            return self.cpu_models[model_key], self.tokenizers[model_key]
                    else:
                        self.logger.warning(f"Using CPU model because GPU memory is insufficient for {model_key}")
                        self.models[model_key] = self.cpu_models[model_key]
                        return self.cpu_models[model_key], self.tokenizers[model_key]
                
                # Check if we're asking for CPU but have GPU version
                if device == "cpu" and current_device == "cuda" and model_key in self.gpu_models:
                    if model_key in self.cpu_models:
                        self.logger.debug(f"Using cached CPU model as requested: {model_key}")
                        self.models[model_key] = self.cpu_models[model_key]
                        return self.cpu_models[model_key], self.tokenizers[model_key]
                    else:
                        # Need to load CPU version
                        self.logger.info(f"Loading CPU version of model that was previously only on GPU: {model_key}")
            
            # If we get here, we need to load the model with our advanced device management
            token = token or self.hf_token
            
            os.environ["HF_HOME"] = self.models_dir
            os.environ["TRANSFORMERS_CACHE"] = os.path.join(self.models_dir, "transformers")
            os.environ["HF_DATASETS_CACHE"] = os.path.join(self.models_dir, "datasets")
            os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(self.models_dir, "hub")
            
            model, tokenizer = self._load_model_with_device_fallback(
                model_name, model_type, device, token, trust_remote_code
            )
            
            self.models[model_key] = model
            self.tokenizers[model_key] = tokenizer
            self.last_used[model_key] = time.time()
            
            return model, tokenizer
            
    def get_model_for_inference(self, model_key: str, batch_size: int = 1):
        """
        Get the appropriate model (GPU or CPU) for inference based on current memory usage.
        This is called during inference to potentially switch from GPU to CPU if GPU memory is critical.
        """
        if model_key not in self.model_locks:
            raise ValueError(f"Model {model_key} is not loaded")
            
        with self.model_locks[model_key]:
            self.last_used[model_key] = time.time()
            
            # If we have GPU and CPU versions, decide which to use based on memory
            if model_key in self.gpu_models and model_key in self.cpu_models:
                if DeviceManager.is_gpu_memory_critical() or not DeviceManager.is_gpu_suitable_for_inference(batch_size):
                    self.logger.warning(f"GPU memory critical or insufficient for batch size {batch_size}, using CPU model for {model_key}")
                    self.models[model_key] = self.cpu_models[model_key]
                    return self.cpu_models[model_key], "cpu"
                else:
                    self.models[model_key] = self.gpu_models[model_key]
                    return self.gpu_models[model_key], "cuda"
            
            # Otherwise return whatever we have
            if model_key in self.gpu_models:
                self.models[model_key] = self.gpu_models[model_key]
                return self.gpu_models[model_key], "cuda"
            elif model_key in self.cpu_models:
                self.models[model_key] = self.cpu_models[model_key]
                return self.cpu_models[model_key], "cpu"
            else:
                raise RuntimeError(f"Model {model_key} not found in either GPU or CPU cache")
    
    def shutdown(self):
        self.running = False
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=1.0)
        
        # Clean up both GPU and CPU versions of all models
        model_keys = list(set(list(self.gpu_models.keys()) + list(self.cpu_models.keys())))
        
        for model_key in model_keys:
            try:
                with self.model_locks.get(model_key, threading.Lock()):
                    if model_key in self.gpu_models:
                        del self.gpu_models[model_key]
                    if model_key in self.cpu_models:
                        del self.cpu_models[model_key]
                    if model_key in self.models:
                        del self.models[model_key]
                    if model_key in self.tokenizers:
                        del self.tokenizers[model_key]
            except Exception as e:
                self.logger.error(f"Error during shutdown, clearing model {model_key}: {e}")
        
        if torch.cuda.is_available():
            DeviceManager.clear_gpu_memory()
        
        self.logger.info("Model cache shutdown complete")