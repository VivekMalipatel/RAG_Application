import os
import logging
import torch
from config import settings

logger = logging.getLogger(__name__)

class DeviceManager:
    """
    Manages device selection and configuration for machine learning models.
    Supports CUDA, MPS (Apple Silicon), and CPU devices.
    """
    
    @staticmethod
    def get_optimal_device() -> str:
        """
        Determines the optimal device based on configuration and availability.
        
        Returns:
            str: The device to use ('cuda', 'mps', or 'cpu')
        """
        # Apply CUDA_VISIBLE_DEVICES from settings if specified
        if settings.CUDA_VISIBLE_DEVICES:
            os.environ["CUDA_VISIBLE_DEVICES"] = settings.CUDA_VISIBLE_DEVICES
            logger.info(f"Setting CUDA_VISIBLE_DEVICES={settings.CUDA_VISIBLE_DEVICES}")
        
        # If specific device is explicitly set
        if settings.PREFERRED_DEVICE != "auto":
            if settings.PREFERRED_DEVICE == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available, falling back to CPU")
                return "cpu"
                
            elif settings.PREFERRED_DEVICE == "mps" and not DeviceManager.is_mps_available():
                if settings.MPS_FALLBACK_TO_CPU:
                    logger.warning("MPS requested but not available, falling back to CPU")
                    return "cpu"
                else:
                    raise RuntimeError("MPS requested but not available, and fallback to CPU is disabled")
                    
            elif settings.PREFERRED_DEVICE in ["cuda", "mps", "cpu"]:
                logger.info(f"Using explicitly configured device: {settings.PREFERRED_DEVICE}")
                return settings.PREFERRED_DEVICE
            else:
                logger.warning(f"Unknown device '{settings.PREFERRED_DEVICE}', falling back to auto detection")
        
        # Auto detect best available device
        if torch.cuda.is_available():
            device = "cuda"
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            # Log available VRAM
            for i in range(torch.cuda.device_count()):
                free_mem = torch.cuda.mem_get_info(i)[0] / (1024**3)  # Convert to GB
                total_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # Convert to GB
                logger.info(f"CUDA:{i} - {torch.cuda.get_device_name(i)} - {free_mem:.2f}GB free of {total_mem:.2f}GB")
        elif DeviceManager.is_mps_available():
            device = "mps"
            logger.info("Using MPS (Apple Silicon) device")
        else:
            device = "cpu"
            logger.info("Using CPU device")
            
        return device
    
    @staticmethod
    def is_mps_available() -> bool:
        """Check if MPS (Metal Performance Shaders) is available on macOS."""
        try:
            return torch.backends.mps.is_available() and torch.backends.mps.is_built()
        except AttributeError:
            # Earlier versions of PyTorch might not have MPS
            return False
            
    @staticmethod
    def get_device_map(device: str = None) -> dict:
        """
        Creates a device map configuration for model loading based on the strategy.
        
        Args:
            device: The device to use ('cuda', 'mps', 'cpu', or None to auto-detect)
            
        Returns:
            dict or str: Device map configuration for model loading
        """
        if device is None:
            device = DeviceManager.get_optimal_device()
            
        # For balanced loading across multiple GPUs
        if device == "cuda" and settings.DEVICE_MAP_STRATEGY == "balanced" and torch.cuda.device_count() > 1:
            return "balanced"
        # For sequential loading (fills up GPU 0 first, then GPU 1, etc.)
        elif device == "cuda" and settings.DEVICE_MAP_STRATEGY == "sequential" and torch.cuda.device_count() > 1:
            return "sequential"
        # For CPU offloading when memory is limited
        elif device == "cuda" and settings.LOW_CPU_MEM_USAGE:
            return {"": 0}  # Map all to first CUDA device with low memory usage
        # For all other cases, use the simple device string
        else:
            return device
            
    @staticmethod
    def get_model_kwargs(device: str = None) -> dict:
        """
        Get appropriate kwargs for model loading based on device and settings.
        
        Args:
            device: The device to use (will auto-detect if None)
            
        Returns:
            dict: Keyword arguments for model loading
        """
        if device is None:
            device = DeviceManager.get_optimal_device()
            
        kwargs = {}
        
        # Handle device mapping
        device_map = DeviceManager.get_device_map(device)
        if isinstance(device_map, dict) or isinstance(device_map, str):
            kwargs["device_map"] = device_map
        else:
            kwargs["device"] = device
            
        # Add memory optimization flags
        if settings.LOW_CPU_MEM_USAGE:
            kwargs["low_cpu_mem_usage"] = True
            
        return kwargs
            
    @staticmethod
    def move_tensors_to_device(data, device: str = None):
        """
        Recursively moves PyTorch tensors to the specified device.
        
        Args:
            data: Data structure containing tensors (can be nested)
            device: Device to move tensors to (defaults to optimal device)
            
        Returns:
            The same data structure with all tensors moved to the device
        """
        if device is None:
            device = DeviceManager.get_optimal_device()
            
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