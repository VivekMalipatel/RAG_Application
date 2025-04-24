import os
import logging
import torch
from config import settings

logger = logging.getLogger(__name__)

class DeviceManager:
    """Manages device selection for machine learning models"""
    
    _cached_device = None
    _device_info_logged = False
    
    @staticmethod
    def get_optimal_device() -> str:
        """Determine the optimal device (cuda, mps, or cpu)"""
        # Return cached device if already determined
        if DeviceManager._cached_device is not None:
            return DeviceManager._cached_device
            
        # Apply CUDA_VISIBLE_DEVICES from settings if specified
        if settings.CUDA_VISIBLE_DEVICES:
            os.environ["CUDA_VISIBLE_DEVICES"] = settings.CUDA_VISIBLE_DEVICES
        
        # Use explicit device configuration if set
        if settings.PREFERRED_DEVICE != "auto":
            device = DeviceManager._check_explicit_device(settings.PREFERRED_DEVICE)
        else:
            # Auto detect best available device
            device = DeviceManager._auto_detect_device()
            
        # Cache the result
        DeviceManager._cached_device = device
        return device
    
    @staticmethod
    def _check_explicit_device(requested_device: str) -> str:
        """Check if explicitly requested device is available"""
        if requested_device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            return "cpu"
            
        elif requested_device == "mps" and not DeviceManager.is_mps_available():
            if settings.MPS_FALLBACK_TO_CPU:
                logger.warning("MPS requested but not available, falling back to CPU")
                return "cpu"
            else:
                logger.error("MPS requested but not available, and fallback to CPU is disabled")
                return "cpu"  # Still fall back to CPU rather than crashing
                
        elif requested_device in ["cuda", "mps", "cpu"]:
            logger.info(f"Using explicitly configured device: {requested_device}")
            return requested_device
        else:
            logger.warning(f"Unknown device '{requested_device}', falling back to auto detection")
            return DeviceManager._auto_detect_device()
    
    @staticmethod
    def _auto_detect_device() -> str:
        """Automatically detect the best available device"""
        if torch.cuda.is_available():
            device = "cuda"
            # Only log detailed device info once
            if not DeviceManager._device_info_logged:
                logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
                try:
                    # Log available VRAM for each GPU
                    for i in range(torch.cuda.device_count()):
                        try:
                            free_mem = torch.cuda.mem_get_info(i)[0] / (1024**3)  # Convert to GB
                            total_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                            logger.info(f"CUDA:{i} - {torch.cuda.get_device_name(i)} - {free_mem:.2f}GB free of {total_mem:.2f}GB")
                        except (RuntimeError, ValueError):
                            # Some systems can't get memory info
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
        """Check if MPS (Metal Performance Shaders) is available"""
        try:
            return torch.backends.mps.is_available() and torch.backends.mps.is_built()
        except (AttributeError, ImportError):
            return False
            
    @staticmethod
    def get_device_map(device: str = None) -> dict:
        """Create device map for model loading"""
        if device is None:
            device = DeviceManager.get_optimal_device()
            
        # Multi-GPU setups
        if device == "cuda" and torch.cuda.device_count() > 1:
            if settings.DEVICE_MAP_STRATEGY == "balanced":
                return "balanced"
            elif settings.DEVICE_MAP_STRATEGY == "sequential":
                return "sequential"
            else:
                # Default to balanced for multi-GPU
                return "balanced"
                
        # Single device setups
        if device == "cuda" and settings.LOW_CPU_MEM_USAGE:
            return {"": 0}  # Map all to first CUDA device with low memory
        
        return device
            
    @staticmethod
    def get_model_kwargs(device: str = None) -> dict:
        """Get kwargs for model loading based on device"""
        if device is None:
            device = DeviceManager.get_optimal_device()
            
        kwargs = {
            "torch_dtype": torch.float16 if device != "cpu" else torch.float32
        }
        
        # Handle device mapping
        device_map = DeviceManager.get_device_map(device)
        if isinstance(device_map, dict) or isinstance(device_map, str):
            kwargs["device_map"] = device_map
        
        # Add memory optimization flags
        if settings.LOW_CPU_MEM_USAGE or device == "cuda":
            kwargs["low_cpu_mem_usage"] = True
            
        return kwargs
            
    @staticmethod
    def move_tensors_to_device(data, device: str = None):
        """Recursively move PyTorch tensors to device"""
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
            return data  # Return original data if movement fails
            
    @staticmethod
    def get_memory_stats() -> dict:
        """Get memory usage statistics"""
        stats = {"cpu_percent": None}
        
        device = DeviceManager.get_optimal_device()
        
        # Get GPU memory stats
        if device == "cuda":
            try:
                current_device = torch.cuda.current_device()
                stats["gpu_allocated"] = torch.cuda.memory_allocated(current_device) / (1024**3)  # GB
                stats["gpu_reserved"] = torch.cuda.memory_reserved(current_device) / (1024**3)  # GB
                
                if hasattr(torch.cuda, "mem_get_info"):
                    free_mem, total_mem = torch.cuda.mem_get_info(current_device)
                    stats["gpu_free"] = free_mem / (1024**3)  # GB
                    stats["gpu_total"] = total_mem / (1024**3)  # GB
            except Exception as e:
                logger.warning(f"Could not get GPU memory stats: {e}")
                
        # Get CPU memory stats
        try:
            import psutil
            stats["cpu_percent"] = psutil.cpu_percent()
            stats["ram_used_percent"] = psutil.virtual_memory().percent
            stats["ram_available_gb"] = psutil.virtual_memory().available / (1024**3)  # GB
        except ImportError:
            pass
            
        return stats