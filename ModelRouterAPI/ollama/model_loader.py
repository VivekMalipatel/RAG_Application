import os
import hashlib
import tempfile
import logging
import asyncio
import subprocess
import aiohttp
import shutil
import threading
import torch
from typing import Optional, Dict, List, Any, Tuple
from aiohttp import ClientTimeout
from pathlib import Path
from safetensors.torch import load_file
from huggingface_hub import snapshot_download, HfApi

from config import settings

logger = logging.getLogger(__name__)

class OllamaModelLoader:
    """
    Native implementation for converting models from Hugging Face to GGUF format
    and loading them into Ollama without requiring a separate service.
    """
    
    # Supported quantization types for GGUF conversion
    GGUF_QUANTIZATION_TYPES = [
        "F32", "F16", "BF16", "Q8_0", "Q4_0", "Q4_1", 
        "Q5_0", "Q5_1", "Q2_K", "Q3_K", "Q3_K_S", 
        "Q3_K_M", "Q3_K_L"
    ]
    
    def __init__(self):
        """
        Initialize the model loader with necessary paths and configuration.
        """
        self.ollama_base_url = settings.OLLAMA_BASE_URL
        self.hf_token = settings.HUGGINGFACE_API_TOKEN
        
        # Ensure temp directories exist
        self.temp_dir = Path(tempfile.gettempdir()) / "model_router_ollama"
        self.models_dir = Path(self.temp_dir) / "models"
        self.downloads_dir = Path(self.temp_dir) / "downloads"
        
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.downloads_dir.mkdir(parents=True, exist_ok=True)
        
        # Path to the GGUF converter script in the local llama.cpp directory
        self.llama_converter_script = Path(__file__).parent / "llama.cpp" / "convert_hf_to_gguf.py"
        
        # Set default headers
        self.headers = {"Content-Type": "application/json"}
        
        # Store ongoing conversion tasks
        self.processing_tasks = {}
        
        logger.info(f"OllamaModelLoader initialized")
    
    async def is_model_available(self, model_name: str) -> bool:
        """
        Check if a specific model is available in Ollama.
        
        Args:
            model_name: The name of the model to check
            
        Returns:
            bool: True if model is available, False otherwise
        """
        url = f"{self.ollama_base_url}/api/tags"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        logger.error(f"Failed to get Ollama models: {await response.text()}")
                        return False
                    
                    data = await response.json()
                    available_models = [model["name"].split(":")[0] for model in data.get("models", [])]
                    logger.info(f"Available models in Ollama: {available_models}")
                    return model_name in available_models
                    
        except Exception as e:
            logger.error(f"Error checking model availability: {str(e)}")
            return False
    
    async def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get a list of all available models in Ollama.
        
        Returns:
            List[Dict[str, Any]]: List of model information dictionaries
        """
        url = f"{self.ollama_base_url}/api/tags"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        logger.error(f"Failed to get Ollama models: {await response.text()}")
                        return []
                    
                    data = await response.json()
                    return data.get("models", [])
                    
        except Exception as e:
            logger.error(f"Error getting available models: {str(e)}")
            return []
    
    async def ensure_model_available(
        self, 
        hf_repo: str, 
        quantization: str = "Q8_0", 
        force_convert: bool = False
    ) -> Optional[str]:
        """
        Ensures a Hugging Face model is available in Ollama by converting it to GGUF if needed.
        
        Args:
            hf_repo: Hugging Face repository ID (e.g., "meta-llama/Llama-3.1-8B-Instruct")
            quantization: Quantization level (e.g., "Q8_0", "Q4_K_M", "Q5_K_M")
            force_convert: Force conversion even if model exists
            
        Returns:
            Optional[str]: Model name if successful, None if failed
        """
        # Validate quantization type
        if quantization not in self.GGUF_QUANTIZATION_TYPES:
            logger.error(f"Invalid quantization type: {quantization}. Must be one of {self.GGUF_QUANTIZATION_TYPES}")
            return None
            
        # Format the model name following Ollama conventions
        model_name = hf_repo.replace("/", "_").lower() + f"_{quantization}"
        
        # Check if model already exists in Ollama
        if not force_convert and await self.is_model_available(model_name):
            logger.info(f"Model {model_name} is already available in Ollama")
            return model_name
        
        # Begin conversion process
        logger.info(f"Converting model {hf_repo} from Hugging Face with quantization {quantization}...")
        
        try:
            # Generate a unique task ID for this conversion
            task_id = f"{hf_repo.replace('/', '_')}_{quantization}"
            self.processing_tasks[task_id] = "Processing"
            
            # Step 1: Download and convert model from Hugging Face
            gguf_path = await self._download_and_convert_model(hf_repo, quantization, task_id)
            if not gguf_path:
                logger.error(f"Failed to convert {hf_repo} to GGUF format")
                self.processing_tasks[task_id] = "Failed"
                return None
            
            # Step 2: Calculate file hash
            file_digest = self._calculate_file_hash(gguf_path)
            
            # Step 3: Upload GGUF to Ollama
            gguf_filename = os.path.basename(gguf_path)
            logger.info(f"Uploading {gguf_filename} to Ollama...")
            success = await self._upload_gguf_to_ollama(gguf_path, file_digest)
            if not success:
                logger.error(f"Failed to upload GGUF file to Ollama")
                self.processing_tasks[task_id] = "Failed"
                return None
            
            # Step 4: Register model in Ollama
            logger.info(f"Registering model {model_name} in Ollama...")
            success = await self._register_model_in_ollama(model_name, gguf_filename, file_digest)
            if not success:
                logger.error(f"Failed to register model {model_name} in Ollama")
                self.processing_tasks[task_id] = "Failed"
                return None
            
            # Update task status
            self.processing_tasks[task_id] = "Completed"
            
            logger.info(f"Successfully registered model {model_name} in Ollama")
            return model_name
            
        except Exception as e:
            logger.error(f"Error ensuring model availability: {str(e)}")
            return None
    
    async def _download_and_convert_model(
        self, 
        hf_repo: str, 
        quantization: str,
        task_id: str
    ) -> Optional[str]:
        """
        Download a model from Hugging Face and convert it to GGUF format.
        
        Args:
            hf_repo: Hugging Face repository ID
            quantization: Quantization level
            task_id: Unique ID for tracking this task
            
        Returns:
            Optional[str]: Path to GGUF file if successful, None if failed
        """
        # Create a unique directory for this model
        model_hash = hashlib.md5(f"{hf_repo}_{quantization}".encode()).hexdigest()
        model_dir = os.path.join(self.models_dir, model_hash)
        os.makedirs(model_dir, exist_ok=True)
        
        # Prepare directories
        local_dir = os.path.join(model_dir, "hf_model")
        os.makedirs(local_dir, exist_ok=True)
        
        # Prepare output file path
        model_name = hf_repo.replace("/", "_").lower()
        gguf_output = os.path.join(self.downloads_dir, f"{model_name}_{quantization}.gguf")
        
        # Check if output file exists already
        if os.path.exists(gguf_output) and not self._is_file_empty(gguf_output):
            logger.info(f"GGUF file already exists at {gguf_output}, skipping conversion")
            return gguf_output
            
        try:
            # Download model from Hugging Face
            self.processing_tasks[task_id] = "Downloading model"
            logger.info(f"Downloading model {hf_repo} from Hugging Face...")
            
            # Use huggingface_hub to download the model in the main thread to avoid issues
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: snapshot_download(
                    repo_id=hf_repo,
                    local_dir=local_dir,
                    local_dir_use_symlinks=False,
                    token=self.hf_token
                )
            )
            
            # Save model download timestamp
            api = HfApi(token=self.hf_token)
            model_info = api.model_info(hf_repo)
            last_modified_remote = model_info.lastModified.isoformat()
            with open(os.path.join(local_dir, ".last_modified"), "w") as f:
                f.write(last_modified_remote)
                
            logger.info(f"Model download complete to {local_dir}")
            
            # Check for and convert safetensors files
            self.processing_tasks[task_id] = "Converting safetensors"
            await self._convert_safetensors_to_pytorch(local_dir)
            
            # Convert to GGUF
            self.processing_tasks[task_id] = "Converting to GGUF"
            success = await self._convert_to_gguf(local_dir, gguf_output, quantization)
            if not success:
                return None
                
            logger.info(f"GGUF conversion complete, file saved at {gguf_output}")
            return gguf_output
                
        except Exception as e:
            logger.error(f"Error in download and convert: {str(e)}")
            return None
    
    def _is_file_empty(self, file_path: str) -> bool:
        """Check if a file is empty or very small (likely corrupted)"""
        try:
            return os.path.getsize(file_path) < 1024  # Less than 1KB is suspicious for a model
        except OSError:
            return True
    
    async def _convert_safetensors_to_pytorch(self, local_dir: str) -> None:
        """
        Convert SafeTensors files to PyTorch format if needed.
        
        Args:
            local_dir: Directory containing the model files
        """
        try:
            for filename in os.listdir(local_dir):
                if filename.endswith(".safetensors"):
                    safetensor_path = os.path.join(local_dir, filename)
                    pytorch_path = safetensor_path.replace(".safetensors", ".bin")
                    if os.path.exists(pytorch_path):
                        logger.info(f"Skipping conversion, {pytorch_path} already exists.")
                        continue
                    
                    logger.info(f"Converting {safetensor_path} to {pytorch_path}...")
                    
                    # Run conversion in an executor to avoid blocking
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(
                        None,
                        lambda: self._convert_single_safetensor(safetensor_path, pytorch_path)
                    )
                    
                    logger.info(f"Converted SafeTensors to PyTorch: {pytorch_path}")
                    
        except Exception as e:
            logger.error(f"Error converting SafeTensors in {local_dir}: {str(e)}")
            raise
    
    def _convert_single_safetensor(self, safetensor_path: str, pytorch_path: str) -> None:
        """Convert a single safetensor file to PyTorch format"""
        tensors = load_file(safetensor_path)
        torch.save(tensors, pytorch_path)
    
    async def _convert_to_gguf(self, local_dir: str, output_file: str, quantization_type: str) -> bool:
        """
        Convert the model to GGUF format using llama.cpp scripts.
        
        Args:
            local_dir: Path to the downloaded model
            output_file: Path to save the GGUF file
            quantization_type: Type of quantization to use
            
        Returns:
            bool: True if conversion was successful, False otherwise
        """
        try:
            logger.info(f"Converting {local_dir} to GGUF...")
            
            # Check for converter script
            if not self.llama_converter_script.exists():
                logger.error(f"Converter script not found at {self.llama_converter_script}")
                return False
                
            converter_script = str(self.llama_converter_script)
            logger.info(f"Using converter script: {converter_script}")
                
            # Prepare conversion command
            command = [
                "python3", converter_script,
                local_dir, 
                "--outfile", output_file, 
                "--outtype", quantization_type.lower()
            ]
            
            # Run conversion process
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                logger.error(f"GGUF conversion failed with return code {process.returncode}")
                logger.error(f"Conversion stdout: {stdout.decode()}")
                logger.error(f"Conversion stderr: {stderr.decode()}")
                if "Model MultiModalityCausalLM is not supported" in stderr.decode():
                    logger.error("The model architecture is not supported for GGUF conversion.")
                return False
                
            if not os.path.exists(output_file) or self._is_file_empty(output_file):
                logger.error(f"GGUF file not created or is empty: {output_file}")
                return False
                
            logger.info(f"Successfully created GGUF file at {output_file}")
            return True
                
        except Exception as e:
            logger.error(f"Error in GGUF conversion: {str(e)}")
            return False
            
    async def _upload_gguf_to_ollama(self, gguf_path: str, file_digest: str) -> bool:
        """
        Upload GGUF file to Ollama.
        
        Args:
            gguf_path: Path to GGUF file
            file_digest: SHA256 hash of the file
            
        Returns:
            bool: True if successful, False otherwise
        """
        url = f"{self.ollama_base_url}/api/blobs/{file_digest}"
        
        try:
            # Use longer timeout for large files
            timeout = ClientTimeout(total=3600)  # 1 hour
            async with aiohttp.ClientSession(timeout=timeout) as session:
                with open(gguf_path, "rb") as f:
                    async with session.post(url, data=f) as response:
                        if response.status not in [200, 201]:
                            logger.error(f"Failed to upload GGUF file: {await response.text()}")
                            return False
                        return True
                        
        except Exception as e:
            logger.error(f"Error uploading GGUF file: {str(e)}")
            return False
            
    async def _register_model_in_ollama(
        self, 
        model_name: str, 
        gguf_filename: str, 
        file_digest: str
    ) -> bool:
        """
        Register model in Ollama.
        
        Args:
            model_name: Name to register the model as
            gguf_filename: Filename of the GGUF file
            file_digest: SHA256 hash of the file
            
        Returns:
            bool: True if successful, False otherwise
        """
        url = f"{self.ollama_base_url}/api/create"
        payload = {
            "name": model_name,
            "modelfile": f"FROM {file_digest}\n",
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=self.headers) as response:
                    if response.status != 200:
                        logger.error(f"Failed to register model: {await response.text()}")
                        return False
                    return True
                    
        except Exception as e:
            logger.error(f"Error registering model: {str(e)}")
            return False
            
    def _calculate_file_hash(self, file_path: str) -> str:
        """
        Calculate SHA256 hash of a file.
        
        Args:
            file_path: Path to file
            
        Returns:
            str: SHA256 hash in format 'sha256:{hash}'
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return f"sha256:{sha256_hash.hexdigest()}"
        
    async def get_task_status(self, task_id: str) -> str:
        """
        Get the status of a model conversion task.
        
        Args:
            task_id: The task ID to check
            
        Returns:
            str: Status of the task
        """
        return self.processing_tasks.get(task_id, "Not Found")