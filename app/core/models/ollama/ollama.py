import aiohttp
import logging
import os
import json
import asyncio
import hashlib
from typing import Optional, List, Dict, Any
from app.config import settings
from aiohttp import ClientTimeout
from pydantic import BaseModel
import requests

class OllamaClient:
    """
    Unified client to interact with Ollama, supporting model retrieval, text generation, and dynamic GGUF conversion.
    """

    def __init__(
        self,
        hf_repo: str,
        quantization: str = "Q8_0",
        system_prompt: Optional[str] = "You are a helpful AI assistant.",
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 1024,
        stream: bool = False,
        **kwargs
    ):
        """
        Initializes an Ollama client with customized model settings.

        Args:
            hf_repo (str): Hugging Face model repository ID (e.g., "meta-llama/Llama-3.1-8B-Instruct").
            quantization (str): Quantization format (default: "Q8_0").
            system_prompt (Optional[str]): Custom system prompt for the model.
            temperature (float): Sampling temperature for randomness.
            top_p (float): Probability mass for nucleus sampling.
            max_tokens (int): Maximum tokens allowed in response.
        """
        self.base_url = settings.OLLAMA_URL
        self.gguf_service_url = settings.GGUF_CONVERTER_URL
        self.hf_repo = hf_repo
        self.quantization = quantization
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.hf_token = settings.HUGGINGFACE_TOKEN
        self.model_name = hf_repo.replace("/", "_") + f"_{quantization}"
        self.headers = {"Content-Type": "application/json"}
        self.stream = stream

    async def generate_text(self, prompt: str, max_tokens: int = None, stream: bool = None) -> str:
        """Generates a response from Ollama LLM."""
        url = f"{self.base_url}/api/chat"
        stream = stream if stream is not None else self.stream

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            "options": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "num_predict": max_tokens if max_tokens else self.max_tokens
            },
            "stream": stream,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=self.headers) as response:
                    if response.status != 200:
                        error_msg = await response.text()
                        logging.error(f"Ollama API Error: {error_msg}")
                        return f"Ollama API Error: {error_msg}"

                    if stream:
                        return await self._handle_stream(response)
                    else:
                        data = await response.json()
                        return data.get("message", {}).get("content", "")

        except Exception as e:
            logging.error(f"Error communicating with Ollama API: {str(e)}")
            return "An error occurred while processing the request."

    async def _handle_stream(self, response) -> str:
        """Handles streaming responses from Ollama."""
        final_response = ""
        async for line in response.content:
            try:
                chunk = json.loads(line.decode("utf-8"))
                final_response += chunk.get("message", {}).get("content", "")
            except Exception as e:
                logging.error(f"Streaming decode error: {str(e)}")
                break
        return final_response
    
    async def generate_structured_output(
        self, prompt: str, schema: BaseModel, max_tokens: int = None, stream: bool = None
    ) -> Dict[str, Any]:
        """
        Generates a structured response from Ollama using a provided JSON schema.

        Args:
            prompt (str): User input prompt.
            schema (BaseModel): Pydantic model defining expected JSON structure.
            max_tokens (int, optional): Maximum tokens in response.
            stream (bool, optional): Enable/disable streaming.

        Returns:
            Dict[str, Any]: Structured response parsed as per schema.
        """

        url = f"{self.base_url}/api/chat"
        stream = stream if stream is not None else self.stream

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            "format": schema.model_json_schema(),  # Pass structured JSON format
            "options": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "num_predict": max_tokens if max_tokens else self.max_tokens
            },
            "stream": stream,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=self.headers) as response:
                    if response.status != 200:
                        error_msg = await response.text()
                        logging.error(f"Ollama API Error: {error_msg}")
                        return {"error": error_msg}

                    data = await response.json()
                    return schema.model_validate_json(data["message"]["content"])

        except Exception as e:
            logging.error(f"Error communicating with Ollama API: {str(e)}")
            return {"error": str(e)}

    def is_model_available(self) -> bool:
        """
        Checks if the requested model is available.
        Uses a synchronous HTTP request to avoid asyncio.run() issues.
        """

        url = f"{self.base_url}/api/tags"
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code != 200:
                logging.error(f"Ollama Model List Fetch Error: {response.text}")
                return False
            data = response.json()
            available_models = [model["name"].split(":")[0] for model in data.get("models", [])]
            if not self.model_name in available_models:
                import nest_asyncio
                nest_asyncio.apply()
                asyncio.run(self.ensure_model_available())
            return True
        except Exception as e:
            logging.error(f"Failed to fetch Ollama models synchronously: {str(e)}")
            return False
    
    async def get_model_list(self) -> List[str]:
        """Fetches available models from Ollama."""
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

    async def is_model_available_async(self) -> bool:
        """
        Asynchronous version of is_model_available.
        Use this when already in an async context.
        """
        available_models = await self.get_model_list()
        return self.model_name in available_models

    async def ensure_model_available(self) -> Optional[str]:
        """Ensures the model is available in Ollama by downloading and registering it."""
        available_models = await self.get_model_list()
        if self.model_name in available_models:
            logging.info(f"Model {self.model_name} is already registered in Ollama.")
            return self.model_name

        # Convert Model
        logging.info(f"Downloading model {self.hf_repo} from Hugging Face with quantization {self.quantization}...")

        conversion_payload = {
            "repo_id": self.hf_repo,
            "quantization": self.quantization,
            "token": self.hf_token
        }

        async with aiohttp.ClientSession() as session:
            response = await session.post(f"{self.gguf_service_url}/convert", json=conversion_payload)
            if response.status != 200:
                error_msg = await response.text()
                logging.error(f"Failed to start conversion: {error_msg}")
                return None

            response_data = await response.json()
            task_id = response_data.get('task_id')
            if not task_id:
                logging.error("No task_id found in response")
                return None
            
            logging.info(f"Conversion task started with ID: {task_id}")

            # Poll for conversion status
            while True:
                await asyncio.sleep(5)
                status_response = await session.get(f"{self.gguf_service_url}/status/{task_id}")
                
                if status_response.status != 200:
                    logging.error("Failed to fetch conversion status")
                    return None

                status_data = await status_response.json()
                if status_data["status"] == "Processing":
                    logging.info("Model is still processing, waiting...")
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
            try:
                async with aiohttp.ClientSession(timeout=ClientTimeout(total=3600)) as upload_session:
                    async with upload_session.post(
                        f"{self.base_url}/api/blobs/{file_digest}",
                        data=open(local_path, "rb")
                    ) as upload_response:

                        if upload_response.status not in [200, 201]:
                            logging.error("Failed to upload GGUF file")
                            return None    

            except asyncio.TimeoutError:
                logging.error("GGUF upload timed out")
                raise TimeoutError("GGUF upload timed out")
            except aiohttp.ClientError as e:
                logging.error(f"Upload failed with error: {str(e)}")
                raise

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