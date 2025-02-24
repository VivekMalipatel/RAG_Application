import aiohttp
import logging
import time
import os
import requests
from typing import Optional, Dict, List
from app.config import settings
import asyncio
import json
import hashlib

# Default Prompt Template (Can be Overridden)
DEFAULT_TEMPLATE = """{{ if .System }}<|start_header_id|>system<|end_header_id|>
{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>
{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>
{{ .Response }}<|eot_id|>"""

# Default Stop Sequences
DEFAULT_STOP_SEQUENCES = ["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"]

class OllamaClient:
    """
    Unified client to interact with Ollama, supporting model retrieval, customization, and dynamic GGUF conversion.
    """

    def __init__(
        self,
        hf_repo: str,
        quantization: str = "Q8_0",
        system_prompt: str = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 1024,
        embedding: bool = False
    ):
        """
        Initializes an Ollama client with customized model settings.

        Args:
            hf_repo (str): Hugging Face model repository ID (e.g., "meta-llama/Llama-3.1-8B-Instruct").
            quantization (str): Quantization format (default: "Q8_0").
            hf_token (Optional[str]): Hugging Face token for private models.
            system_prompt (Optional[str]): Custom system prompt for the model.
            template (str): Customizable template for structuring prompts.
            stop_sequences (List[str]): Custom stop sequences for formatting control.
            temperature (float): Sampling temperature for randomness.
            top_p (float): Probability mass for nucleus sampling.
            max_tokens (int): Maximum tokens allowed in response.
            embedding (bool): Whether to use the model for embeddings.
        """
        self.base_url = settings.OLLAMA_URL
        self.gguf_service_url = settings.GGUF_CONVERTER_URL
        self.hf_repo = hf_repo
        self.quantization = quantization
        self.hf_token = settings.HUGGINGFACE_TOKEN
        self.system_prompt = system_prompt
        self.template = DEFAULT_TEMPLATE
        self.stop_sequences = DEFAULT_STOP_SEQUENCES
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.embedding = embedding
        self.model_name = hf_repo.replace("/", "_") + f"_{quantization}"
        self.headers = {"Content-Type": "application/json"}
        self.ensure_model_available()

    async def generate(self, prompt: str, max_tokens: int = None, stream: bool = False) -> str:
        """Generates a response from Ollama LLM."""
        url = f"{self.base_url}/api/generate"
        formatted_prompt = self._apply_template(prompt)

        payload = {
            "model": self.model_name,
            "prompt": formatted_prompt,
            "options": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "num_predict": max_tokens if max_tokens else self.max_tokens,
                "stop": self.stop_sequences
            },
            "stream": stream,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=self.headers) as response:
                    if response.status != 200:
                        logging.error(f"Ollama API Error: {await response.text()}")
                        return "Ollama API Error: "+ await response.text()

                    if stream:
                        return await self._handle_stream(response)
                    else:
                        data = await response.json()
                        return data.get("response", "")

        except Exception as e:
            logging.error(f"Error communicating with Ollama API: {str(e)}")
            return "An error occurred while processing the request."

    async def _handle_stream(self, response) -> str:
        """Handles streaming responses from Ollama."""
        final_response = ""
        async for line in response.content:
            try:
                chunk = json.loads(line.decode("utf-8"))
                final_response += chunk.get("response", "")
            except Exception as e:
                logging.error(f"Streaming decode error: {str(e)}")
                break
        return final_response

    def get_model_list(self) -> List[str]:
        """Fetch available models from Ollama server."""
        url = f"{self.base_url}/api/tags"
        try:
            response = requests.get(url, headers=self.headers)
            if response.status_code != 200:
                logging.error(f"Ollama Model List Fetch Error: {response.text}")
                return []
            data = response.json()
            # Strip ':latest' suffix from model names for comparison
            return [model["name"].split(":")[0] for model in data.get("models", [])]
        except Exception as e:
            logging.error(f"Failed to fetch Ollama models: {str(e)}")
            return []

    def ensure_model_available(self) -> Optional[str]:
        """Ensures the model is available in Ollama by uploading and registering it."""
        available_models = self.get_model_list()
        if self.model_name in available_models:
            logging.info(f"Model {self.model_name} is already registered in Ollama.")
            return self.model_name

        # Convert Model
        logging.info(f"Requesting GGUF conversion for {self.hf_repo}...")
        conversion_payload = {
            "repo_id": self.hf_repo,
            "quantization": self.quantization,
            "token": self.hf_token
        }
        
        response = requests.post(
            f"{self.gguf_service_url}/convert",
            json=conversion_payload
        )
        if response.status_code != 200:
            logging.error(f"Failed to start conversion: {response.text}")
            return None

        # Poll for conversion status
        try:
            response_content = response.content.decode('utf-8')
            response_data = json.loads(response_content)
            task_id = response_data.get('task_id')
            
            if not task_id:
                logging.error("No task_id found in response")
                return None
                
            logging.info(f"Got conversion task_id: {task_id}")
            
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse response JSON: {str(e)}")
            return None
        except UnicodeDecodeError as e:
            logging.error(f"Failed to decode response content: {str(e)}")
            return None
        while True:
            time.sleep(5)  # Replace asyncio.sleep with time.sleep
            status_response = requests.get(f"{self.gguf_service_url}/status/{task_id}")
            
            if status_response.status_code != 200:
                logging.error("Failed to fetch conversion status")
                return None
                
            status_data = status_response.json()
            if status_data["status"] == "Processing":
                logging.info("Still processing, waiting...")
                continue
            elif status_data["status"] == "Failed":
                logging.error("Model conversion failed.")
                return None
            else:
                logging.info("Conversion complete!")
                break

        # Upload GGUF file
        # Download GGUF file
        os.makedirs("./models", exist_ok=True)  # Ensure models directory exists
        local_path = f"./models/{task_id}.gguf"
        
        logging.info(f"Downloading GGUF file from {status_data['status']}...")
        response = requests.get(status_data['status'], stream=True)
        if response.status_code != 200:
            logging.error("Failed to download GGUF file")
            return None
            
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        file_digest = self._calculate_file_hash(local_path)
        
        with open(local_path, "rb") as f:
            upload_response = requests.post(
                f"{self.base_url}/api/blobs/{file_digest}",
                data=f
            )
            if upload_response.status_code != 201 and upload_response.status_code != 200:
                logging.error("Failed to upload GGUF file")
                return None

        # Register Model
        create_payload = {
            "model": self.model_name,
            "files": {f"{task_id}.gguf": file_digest}
        }
        
        create_response = requests.post(
            f"{self.base_url}/api/create",
            json=create_payload,
            headers=self.headers
        )
        
        if create_response.status_code == 200:
            logging.info(f"Model {self.model_name} successfully registered in Ollama!")
            try:
                os.remove(local_path)
                logging.info(f"Cleaned up temporary file: {local_path}")
            except Exception as e:
                logging.warning(f"Failed to clean up temporary file: {str(e)}")
            return self.model_name
        else:
            logging.error("Failed to register GGUF model in Ollama.")
            return None
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculates SHA256 hash of the file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return f"sha256:{sha256_hash.hexdigest()}"

    def _apply_template(self, user_prompt: str) -> str:
        """Applies a template to structure the prompt dynamically."""
        return self.template.replace("{{ .System }}", self.system_prompt or "").replace("{{ .Prompt }}", user_prompt).replace("{{ .Response }}", "")