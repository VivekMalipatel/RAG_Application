import os
import logging
import torch
from config import settings
import gc

logger = logging.getLogger(__name__)

class DeviceManager:
    _cached_device = None
    _device_info_logged = False
    # Default thresholds
    GPU_MEMORY_THRESHOLD = 0.8  # 80%
    GPU_MEMORY_CRITICAL = 0.98  # 98%
    CPU_MEMORY_THRESHOLD = 0.8  # 80%
    
    @staticmethod
    def get_optimal_device() -> str:
        if DeviceManager._cached_device is not None:
            return DeviceManager._cached_device
            
        if settings.CUDA_VISIBLE_DEVICES:
            os.environ["CUDA_VISIBLE_DEVICES"] = settings.CUDA_VISIBLE_DEVICES
        
        if settings.PREFERRED_DEVICE != "auto":
            device = DeviceManager._check_explicit_device(settings.PREFERRED_DEVICE)
        else:
            device = DeviceManager._auto_detect_device()
            
        DeviceManager._cached_device = device
        return device
    
    @staticmethod
    def _check_explicit_device(requested_device: str) -> str:
        if requested_device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            return "cpu"
            
        elif requested_device == "mps" and not DeviceManager.is_mps_available():
            if settings.MPS_FALLBACK_TO_CPU:
                logger.warning("MPS requested but not available, falling back to CPU")
                return "cpu"
            else:
                logger.error("MPS requested but not available, and fallback to CPU is disabled")
                return "cpu"
                
        elif requested_device in ["cuda", "mps", "cpu"]:
            logger.info(f"Using explicitly configured device: {requested_device}")
            return requested_device
        else:
            logger.warning(f"Unknown device '{requested_device}', falling back to auto detection")
            return DeviceManager._auto_detect_device()
    
    @staticmethod
    def _auto_detect_device() -> str:
        if torch.cuda.is_available():
            device = "cuda"
            if not DeviceManager._device_info_logged:
                logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
                try:
                    for i in range(torch.cuda.device_count()):
                        try:
                            free_mem = torch.cuda.mem_get_info(i)[0] / (1024**3)
                            total_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                            logger.info(f"CUDA:{i} - {torch.cuda.get_device_name(i)} - {free_mem:.2f}GB free of {total_mem:.2f}GB")
                        except (RuntimeError, ValueError):
                            logger.info(f"CUDA:{i} - {torch.cuda.get_device_name(i)} - Memory info unavailable")
                except Exception as e:
                    logger.warning(f"Could not fetch detailed CUDA device info: {e}")
                DeviceManager._device_info_logged = True
        elif DeviceManager.is_mps_available():
            device = "mps"
            logger.info("Using MPS (Apple Silicon) device")
        else:
            device = "cpu"
            logger.info("Using CPU device")
            
        return device
    
    @staticmethod
    def is_mps_available() -> bool:
        try:
            return torch.backends.mps.is_available() and torch.backends.mps.is_built()
        except (AttributeError, ImportError):
            return False
            
    @staticmethod
    def get_device_map(device: str = None) -> dict:
        if device is None:
            device = DeviceManager.get_optimal_device()
            
        if device == "cuda" and torch.cuda.device_count() > 1:
            if settings.DEVICE_MAP_STRATEGY == "balanced":
                return "balanced"
            elif settings.DEVICE_MAP_STRATEGY == "sequential":
                return "sequential"
            else:
                return "balanced"
                
        if device == "cuda" and settings.LOW_CPU_MEM_USAGE:
            return {"": 0}
        
        return device
            
    @staticmethod
    def get_model_kwargs(device: str = None) -> dict:
        if device is None:
            device = DeviceManager.get_optimal_device()
            
        kwargs = {
            "torch_dtype": torch.float16 if device != "cpu" else torch.float32
        }
        
        device_map = DeviceManager.get_device_map(device)
        if isinstance(device_map, dict) or isinstance(device_map, str):
            kwargs["device_map"] = device_map
        
        if settings.LOW_CPU_MEM_USAGE or device == "cuda":
            kwargs["low_cpu_mem_usage"] = True
            
        return kwargs
            
    @staticmethod
    def move_tensors_to_device(data, device: str = None):
        if device is None:
            device = DeviceManager.get_optimal_device()
            
        try:
            if isinstance(data, torch.Tensor):
                return data.to(device)
            elif isinstance(data, dict):
                return {k: DeviceManager.move_tensors_to_device(v, device) for k, v in data.items()}
            elif isinstance(data, list):
                return [DeviceManager.move_tensors_to_device(item, device) for item in data]
            elif isinstance(data, tuple):
                return tuple(DeviceManager.move_tensors_to_device(item, device) for item in data)
            else:
                return data
        except Exception as e:
            logger.warning(f"Error moving tensor to device {device}: {e}")
            return data
            
    @staticmethod
    def get_memory_stats() -> dict:
        stats = {"cpu_percent": None}
        
        device = DeviceManager.get_optimal_device()
        
        if device == "cuda":
            try:
                current_device = torch.cuda.current_device()
                stats["gpu_allocated"] = torch.cuda.memory_allocated(current_device) / (1024**3)
                stats["gpu_reserved"] = torch.cuda.memory_reserved(current_device) / (1024**3)
                
                if hasattr(torch.cuda, "mem_get_info"):
                    free_mem, total_mem = torch.cuda.mem_get_info(current_device)
                    stats["gpu_free"] = free_mem / (1024**3)
                    stats["gpu_total"] = total_mem / (1024**3)
                    stats["gpu_used_percent"] = 100 - (free_mem / total_mem * 100)
            except Exception as e:
                logger.warning(f"Could not get GPU memory stats: {e}")
                
        try:
            import psutil
            stats["cpu_percent"] = psutil.cpu_percent()
            stats["ram_used_percent"] = psutil.virtual_memory().percent
            stats["ram_available_gb"] = psutil.virtual_memory().available / (1024**3)
            stats["ram_total_gb"] = psutil.virtual_memory().total / (1024**3)
        except ImportError:
            pass
            
        return stats
    
    @staticmethod
    def check_cuda_availability_with_memory_threshold():
        """Check if CUDA is available and has enough memory to load models."""
        if not torch.cuda.is_available():
            return False, "CUDA not available"
            
        try:
            stats = DeviceManager.get_memory_stats()
            if "gpu_used_percent" in stats:
                gpu_usage = stats["gpu_used_percent"] / 100
                if gpu_usage < DeviceManager.GPU_MEMORY_THRESHOLD:
                    return True, f"GPU has sufficient memory (using {gpu_usage*100:.1f}% of total)"
                else:
                    return False, f"GPU memory usage too high: {gpu_usage*100:.1f}% (threshold: {DeviceManager.GPU_MEMORY_THRESHOLD*100}%)"
            else:
                # If we can't get memory info, assume it's OK if CUDA is available
                return True, "CUDA available but couldn't determine memory usage"
        except Exception as e:
            logger.warning(f"Error checking GPU memory: {e}")
            return False, f"Error checking GPU memory: {str(e)}"
    
    @staticmethod
    def check_cpu_memory_availability():
        """Check if the system has enough CPU memory to load models."""
        try:
            import psutil
            mem_info = psutil.virtual_memory()
            if mem_info.percent < DeviceManager.CPU_MEMORY_THRESHOLD * 100:
                return True, f"CPU has sufficient memory (using {mem_info.percent:.1f}%)"
            else:
                return False, f"CPU memory usage too high: {mem_info.percent:.1f}% (threshold: {DeviceManager.CPU_MEMORY_THRESHOLD*100}%)"
        except Exception as e:
            logger.warning(f"Error checking CPU memory: {e}")
            return False, f"Error checking CPU memory: {str(e)}"
    
    @staticmethod
    def is_gpu_memory_critical():
        """Check if GPU memory usage is critically high."""
        if not torch.cuda.is_available():
            return False
            
        try:
            stats = DeviceManager.get_memory_stats()
            if "gpu_used_percent" in stats:
                gpu_usage = stats["gpu_used_percent"] / 100
                return gpu_usage > DeviceManager.GPU_MEMORY_CRITICAL
            return False
        except Exception:
            return False
    
    @staticmethod
    def is_gpu_suitable_for_inference(batch_size=1, estimated_memory_per_item=0.1):
        """
        Check if GPU is suitable for inference based on current memory usage and batch size.
        estimated_memory_per_item is in GB.
        """
        if not torch.cuda.is_available():
            return False
            
        try:
            stats = DeviceManager.get_memory_stats()
            if "gpu_free" in stats:
                required_memory = batch_size * estimated_memory_per_item
                return stats["gpu_free"] >= required_memory
            return True  # If we can't determine, assume it's OK
        except Exception:
            return True  # If we can't determine, assume it's OK
    
    @staticmethod
    def clear_gpu_memory():
        """Force clear the GPU memory cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("GPU memory cache cleared")