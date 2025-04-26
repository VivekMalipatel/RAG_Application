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
    GGUF_QUANTIZATION_TYPES = [
        "F32", "F16", "BF16", "Q8_0", "Q4_0", "Q4_1", 
        "Q5_0", "Q5_1", "Q2_K", "Q3_K", "Q3_K_S", 
        "Q3_K_M", "Q3_K_L"
    ]
    
    def __init__(self):
        self.ollama_base_url = settings.OLLAMA_BASE_URL
        self.hf_token = settings.HUGGINGFACE_API_TOKEN
        
        self.temp_dir = Path(tempfile.gettempdir()) / "model_router_ollama"
        self.models_dir = Path(self.temp_dir) / "models"
        self.downloads_dir = Path(self.temp_dir) / "downloads"
        
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.downloads_dir.mkdir(parents=True, exist_ok=True)
        
        self.llama_converter_script = Path(__file__).parent / "llama.cpp" / "convert_hf_to_gguf.py"
        
        self.headers = {"Content-Type": "application/json"}
        
        self.processing_tasks = {}
        
        logger.info(f"OllamaModelLoader initialized")
    
    async def is_model_available(self, model_name: str) -> bool:
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
        if quantization not in self.GGUF_QUANTIZATION_TYPES:
            logger.error(f"Invalid quantization type: {quantization}. Must be one of {self.GGUF_QUANTIZATION_TYPES}")
            return None
            
        model_name = hf_repo.replace("/", "_").lower() + f"_{quantization}"
        
        if not force_convert and await self.is_model_available(model_name):
            logger.info(f"Model {model_name} is already available in Ollama")
            return model_name
        
        logger.info(f"Converting model {hf_repo} from Hugging Face with quantization {quantization}...")
        
        try:
            task_id = f"{hf_repo.replace('/', '_')}_{quantization}"
            self.processing_tasks[task_id] = "Processing"
            
            gguf_path = await self._download_and_convert_model(hf_repo, quantization, task_id)
            if not gguf_path:
                logger.error(f"Failed to convert {hf_repo} to GGUF format")
                self.processing_tasks[task_id] = "Failed"
                return None
            
            file_digest = self._calculate_file_hash(gguf_path)
            
            gguf_filename = os.path.basename(gguf_path)
            logger.info(f"Uploading {gguf_filename} to Ollama...")
            success = await self._upload_gguf_to_ollama(gguf_path, file_digest)
            if not success:
                logger.error(f"Failed to upload GGUF file to Ollama")
                self.processing_tasks[task_id] = "Failed"
                return None
            
            logger.info(f"Registering model {model_name} in Ollama...")
            success = await self._register_model_in_ollama(model_name, gguf_filename, file_digest)
            if not success:
                logger.error(f"Failed to register model {model_name} in Ollama")
                self.processing_tasks[task_id] = "Failed"
                return None
            
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
        model_hash = hashlib.md5(f"{hf_repo}_{quantization}".encode()).hexdigest()
        model_dir = os.path.join(self.models_dir, model_hash)
        os.makedirs(model_dir, exist_ok=True)
        
        local_dir = os.path.join(model_dir, "hf_model")
        os.makedirs(local_dir, exist_ok=True)
        
        model_name = hf_repo.replace("/", "_").lower()
        gguf_output = os.path.join(self.downloads_dir, f"{model_name}_{quantization}.gguf")
        
        if os.path.exists(gguf_output) and not self._is_file_empty(gguf_output):
            logger.info(f"GGUF file already exists at {gguf_output}, skipping conversion")
            return gguf_output
            
        try:
            self.processing_tasks[task_id] = "Downloading model"
            logger.info(f"Downloading model {hf_repo} from Hugging Face...")
            
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
            
            api = HfApi(token=self.hf_token)
            model_info = api.model_info(hf_repo)
            last_modified_remote = model_info.lastModified.isoformat()
            with open(os.path.join(local_dir, ".last_modified"), "w") as f:
                f.write(last_modified_remote)
                
            logger.info(f"Model download complete to {local_dir}")
            
            self.processing_tasks[task_id] = "Converting safetensors"
            await self._convert_safetensors_to_pytorch(local_dir)
            
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
        try:
            return os.path.getsize(file_path) < 1024
        except OSError:
            return True
    
    async def _convert_safetensors_to_pytorch(self, local_dir: str) -> None:
        try:
            for filename in os.listdir(local_dir):
                if filename.endswith(".safetensors"):
                    safetensor_path = os.path.join(local_dir, filename)
                    pytorch_path = safetensor_path.replace(".safetensors", ".bin")
                    if os.path.exists(pytorch_path):
                        logger.info(f"Skipping conversion, {pytorch_path} already exists.")
                        continue
                    
                    logger.info(f"Converting {safetensor_path} to {pytorch_path}...")
                    
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
        tensors = load_file(safetensor_path)
        torch.save(tensors, pytorch_path)
    
    async def _convert_to_gguf(self, local_dir: str, output_file: str, quantization_type: str) -> bool:
        try:
            logger.info(f"Converting {local_dir} to GGUF...")
            
            if not self.llama_converter_script.exists():
                logger.error(f"Converter script not found at {self.llama_converter_script}")
                return False
                
            converter_script = str(self.llama_converter_script)
            logger.info(f"Using converter script: {converter_script}")
                
            command = [
                "python3", converter_script,
                local_dir, 
                "--outfile", output_file, 
                "--outtype", quantization_type.lower()
            ]
            
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
        url = f"{self.ollama_base_url}/api/blobs/{file_digest}"
        
        try:
            timeout = ClientTimeout(total=3600)
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
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return f"sha256:{sha256_hash.hexdigest()}"
        
    async def get_task_status(self, task_id: str) -> str:
        return self.processing_tasks.get(task_id, "Not Found")