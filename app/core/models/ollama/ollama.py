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

    async def get_model_list(self) -> List[str]:
        """Async method to fetch available models from Ollama server."""
        url = f"{self.base_url}/api/tags"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers) as response:
                    if response.status != 200:
                        logging.error(f"Ollama Model List Fetch Error: {await response.text()}")
                        return []
                    data = await response.json()
                    return [model["name"].split(":")[0] for model in data.get("models", [])]
        except Exception as e:
            logging.error(f"Failed to fetch Ollama models: {str(e)}")
            return []

    async def ensure_model_available(self) -> Optional[str]:
        """Ensures the model is available in Ollama by uploading and registering it."""
        available_models = await self.get_model_list()
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

        async with aiohttp.ClientSession() as session:
            response = await session.post(f"{self.gguf_service_url}/convert", json=conversion_payload)
            if response.status != 200:
                logging.error(f"Failed to start conversion: {await response.text()}")
                return None

            response_data = await response.json()
            task_id = response_data.get('task_id')
            if not task_id:
                logging.error("No task_id found in response")
                return None
            
            logging.info(f"Got conversion task_id: {task_id}")

            # Poll for conversion status
            while True:
                await asyncio.sleep(5)
                status_response = await session.get(f"{self.gguf_service_url}/status/{task_id}")
                
                if status_response.status != 200:
                    logging.error("Failed to fetch conversion status")
                    return None

                status_data = await status_response.json()
                if status_data["status"] == "Processing":
                    logging.info("Still processing, waiting...")
                    continue
                elif status_data["status"] == "Failed":
                    logging.error("Model conversion failed.")
                    return None
                else:
                    logging.info("Conversion complete!")
                    break

            # Download GGUF file
            os.makedirs("./models", exist_ok=True)
            local_path = f"./models/{task_id}.gguf"

            logging.info(f"Downloading GGUF file from {status_data['status']}...")
            file_response = await session.get(status_data['status'])
            if file_response.status != 200:
                logging.error("Failed to download GGUF file")
                return None

            with open(local_path, 'wb') as f:
                f.write(await file_response.read())

            file_digest = self._calculate_file_hash(local_path)

            # Upload GGUF file
            async with aiohttp.ClientSession() as upload_session:
                async with upload_session.post(
                    f"{self.base_url}/api/blobs/{file_digest}",
                    data=open(local_path, "rb")
                ) as upload_response:

                    if upload_response.status not in [200, 201]:
                        logging.error("Failed to upload GGUF file")
                        return None

            # Register Model
            create_payload = {
                "model": self.model_name,
                "files": {f"{task_id}.gguf": file_digest}
            }

            async with aiohttp.ClientSession() as register_session:
                async with register_session.post(
                    f"{self.base_url}/api/create",
                    json=create_payload,
                    headers=self.headers
                ) as create_response:

                    if create_response.status == 200:
                        logging.info(f"Model {self.model_name} successfully registered in Ollama!")
                        os.remove(local_path)  # Clean up temp file
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