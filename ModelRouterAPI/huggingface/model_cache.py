import logging
import threading
import os
import importlib.util
import traceback
from typing import Dict, Tuple, Any, Optional
import time
import torch
from transformers import AutoTokenizer, AutoModel
from model_type import ModelType
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers import AutoTokenizer, AutoConfig, AutoModel
import torch

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
                    if model_key in self.models:
                        self.logger.info(f"Removing idle model: {model_key}")
                        del self.models[model_key]
                        if model_key in self.tokenizers:
                            del self.tokenizers[model_key]
                        if model_key in self.last_used:
                            del self.last_used[model_key]
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
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
            
            if model_key in self.models and model_key in self.tokenizers:
                self.logger.debug(f"Using cached model: {model_key}")
                return self.models[model_key], self.tokenizers[model_key]
            
            self.logger.info(f"Loading model: {model_name} ({model_type.value}) on {device}")
            
            try:
                token = token or self.hf_token

                offload_folder = os.path.join(self.models_dir, "offload", model_key)
                os.makedirs(offload_folder, exist_ok=True)
                
                os.environ["HF_HOME"] = self.models_dir
                os.environ["TRANSFORMERS_CACHE"] = os.path.join(self.models_dir, "transformers")
                os.environ["HF_DATASETS_CACHE"] = os.path.join(self.models_dir, "datasets")
                os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(self.models_dir, "hub")
                
                if self._is_nomic_multimodal_model(model_name):
                    self.logger.info(f"Loading Nomic multimodal model using colpali: {model_name}")
                    try:
                        from transformers.utils.import_utils import is_flash_attn_2_available
                        
                        ModelClass, ProcessorClass = self._get_colpali_class(model_name)
                        
                        model_kwargs = {
                            "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                            "device_map": device,
                            "cache_dir": os.path.join(self.models_dir, "transformers"),
                            "local_files_only": False
                        }
                        
                        if is_flash_attn_2_available():
                            model_kwargs["attn_implementation"] = "flash_attention_2"
                            
                        model = ModelClass.from_pretrained(
                            model_name,
                            **model_kwargs
                        ).eval()
                        
                        processor = ProcessorClass.from_pretrained(
                            model_name, 
                            cache_dir=os.path.join(self.models_dir, "transformers")
                        )
                        
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
                    self.logger.info(f"Loading standard Hugging Face model: {model_name}")
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_name, 
                        token=token,
                        cache_dir=os.path.join(self.models_dir, "transformers")
                    )

                    config = AutoConfig.from_pretrained(
                        model_name,
                        token=token,
                        trust_remote_code=trust_remote_code,
                        cache_dir=os.path.join(self.models_dir, "transformers")
                    )

                    with init_empty_weights():
                        model = AutoModel.from_config(config)

                    model = load_checkpoint_and_dispatch(
                                model,
                                model_name,
                                device_map="auto",
                                offload_folder=offload_folder,
                                dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                            ).eval()
                    
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
        self.running = False
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=1.0)
        
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
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.logger.info("Model cache shutdown complete")