import logging
import json
import os
from typing import Optional, List, Union, AsyncGenerator, Dict, Any
from pydantic import BaseModel
from openai import AsyncOpenAI, APIError
from app.config import settings
import asyncio

class OpenAIClient:
    """
    Unified OpenAI client supporting:
    - Text generation (GPT-4, GPT-3.5, DeepSeek)
    - Text embeddings (text-embedding-ada-002)
    - Image generation (DALL·E)
    - Audio transcription (Whisper)
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        system_prompt: Optional[str] = "You are a helpful AI assistant.",
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = None,
        stream: bool = False
    ):
        """
        Initializes an OpenAI client with customized model settings.

        Args:
            model_name (str): Model name (e.g., "gpt-4-turbo", "text-embedding-ada-002").
            system_prompt (Optional[str]): Custom system instructions.
            temperature (float): Controls randomness in generation.
            top_p (float): Probability mass for nucleus sampling.
            max_tokens (int): Maximum response length.
            frequency_penalty (float): Penalize repeating words.
            presence_penalty (float): Encourage diversity.
            repetition_penalty (float): Adjust repetition tendency.
            image_quality (str): Image quality setting for DALL·E.
            image_style (str): Image style setting for DALL·E.
        """
        self.client = AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_API_URL
        )
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.stream = stream

        # Ensure the model is available before proceeding
        if not self.is_model_available():
            raise ValueError(f"Model {self.model_name} is not supported by OpenAI.")

    async def generate_text(self, prompt: str, max_tokens: Optional[int] = None, stream: bool = None) -> Union[str, AsyncGenerator[str, None]]:
        """
        Generates a response using OpenAI models (GPT-4, GPT-3.5, DeepSeek).
        """
        stream = stream if stream is not None else self.stream
        
        if stream:
            return self._generate_stream(prompt, max_tokens)
        else:
            return await self._generate_complete(prompt, max_tokens)

    async def _generate_stream(self, prompt: str, max_tokens: Optional[int] = None) -> AsyncGenerator[str, None]:
        """Helper method for streaming text generation."""
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": max_tokens if max_tokens else self.max_tokens,
            "stream": True
        }
        
        try:
            response = await self.client.chat.completions.create(**payload)
            async for chunk in response:
                yield chunk.choices[0].delta.content if chunk.choices[0].delta.content else ""
        except Exception as e:
            logging.error(f"Error: {str(e)}")
            yield f"Error: {str(e)}"

    async def _generate_complete(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """Helper method for non-streaming text generation."""
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": max_tokens if max_tokens else self.max_tokens,
            "stream": False
        }
        
        try:
            response = await self.client.chat.completions.create(**payload)
            return response.choices[0].message.content
        except APIError as e:
            logging.error(f"OpenAI API Error: {str(e)}")
            return f"API Error: {str(e)}"
        except Exception as e:
            logging.error(f"Unexpected Error: {str(e)}")
            return "Error processing request"
    
    async def generate_structured_output(
    self, prompt: str, schema: BaseModel, max_tokens: Optional[int] = None, stream: bool = None
) -> Dict[str, Any]:
        """
        Generates a structured response from OpenAI using a provided JSON schema.

        Args:
            prompt (str): User input prompt.
            schema (BaseModel): Pydantic model defining expected JSON structure.
            max_tokens (int, optional): Maximum tokens in response.
            stream (bool, optional): Enable/disable streaming.

        Returns:
            Dict[str, Any]: Structured response parsed as per schema.
        """
        # Don't use streaming for structured outputs as it complicates parsing
        if stream:
            logging.warning("Streaming not supported for structured outputs, falling back to non-streaming")
        
        try:
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_tokens": max_tokens if max_tokens else self.max_tokens,
                "response_format": schema
            }
            
            response = await self.client.chat.completions.create(**payload)
            response_content = response.choices[0].message.content
            
            # Parse and validate response against schema
            try:
                parsed_json = json.loads(response_content)
                return schema.model_validate(parsed_json)
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse response as JSON: {e}")
                return {"error": "Invalid JSON response"}
            except Exception as e:
                logging.error(f"Schema validation error: {e}")
                return {"error": str(e)}
                
        except APIError as e:
            logging.error(f"OpenAI API Error: {str(e)}")
            return {"error": str(e)}
        except Exception as e:
            logging.error(f"Unexpected Error: {str(e)}")
            return {"error": str(e)}

    async def get_model_list(self) -> List[str]:
        """
        Retrieves available models from OpenAI.
        """
        try:
            models = await self.client.models.list()
            return [model.id for model in models.data]
        except APIError as e:
            logging.error(f"Model List Error: {str(e)}")
            return []

    def is_model_available(self) -> bool:
        """
        Checks if the requested model is available in OpenAI.
        """
        import nest_asyncio
        nest_asyncio.apply()
        available_models = asyncio.run(self.get_model_list())
        return self.model_name in available_models

    def set_system_prompt(self, system_prompt: str):
        """Updates the system prompt dynamically."""
        self.system_prompt = system_prompt
        logging.info(f"System prompt updated for OpenAI model {self.model_name}: {system_prompt}")

    async def generate_text_batch(self, prompts: List[Dict[str, Any]], model_name: Optional[str] = None, 
                                 system_prompt: Optional[str] = None, max_tokens: Optional[int] = None) -> str:
        """
        Prepares and submits a batch of text generation requests for processing.
        
        Args:
            prompts: List of dictionaries with 'custom_id' and 'content' keys
            model_name: Optional model name to override the default
            system_prompt: Optional system prompt to override the default
            max_tokens: Optional max tokens to override the default
            
        Returns:
            Batch ID for tracking the batch job
        """
        current_model = model_name or self.model_name
        current_system = system_prompt or self.system_prompt
        current_max_tokens = max_tokens or self.max_tokens
        
        # Create batch input file
        batch_file_path = await self._create_batch_input_file(prompts, current_model, current_system, current_max_tokens)
        
        try:
            # Upload batch file
            file_id = await self._upload_batch_file(batch_file_path)
            
            # Create batch job
            batch = await self._create_batch(file_id)
            
            # Clean up temporary file
            if os.path.exists(batch_file_path):
                os.remove(batch_file_path)
                
            return batch.id
        except Exception as e:
            logging.error(f"Batch processing error: {str(e)}")
            # Clean up temporary file on error
            if os.path.exists(batch_file_path):
                os.remove(batch_file_path)
            raise

    async def _create_batch_input_file(self, prompts: List[Dict[str, Any]], model_name: str, 
                                     system_prompt: str, max_tokens: int) -> str:
        """Creates a JSONL file for batch processing with OpenAI."""
        temp_file_path = f"batch_input_{id(prompts)}.jsonl"
        
        with open(temp_file_path, "w") as f:
            for prompt_data in prompts:
                if "custom_id" not in prompt_data or "content" not in prompt_data:
                    raise ValueError("Each prompt must have 'custom_id' and 'content' keys")
                    
                batch_item = {
                    "custom_id": prompt_data["custom_id"],
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model_name,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt_data["content"]}
                        ],
                        "max_tokens": max_tokens,
                        "temperature": self.temperature,
                        "top_p": self.top_p
                    }
                }
                f.write(json.dumps(batch_item) + "\n")
                
        return temp_file_path

    async def _upload_batch_file(self, file_path: str) -> str:
        """Uploads the batch input file using the Files API."""
        try:
            with open(file_path, "rb") as file:
                response = await self.client.files.create(
                    file=file,
                    purpose="batch"
                )
            return response.id
        except Exception as e:
            logging.error(f"Error uploading batch file: {str(e)}")
            raise

    async def _create_batch(self, file_id: str, metadata: Dict[str, str] = None) -> Any:
        """Creates a batch processing job."""
        try:
            batch = await self.client.batches.create(
                input_file_id=file_id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata=metadata
            )
            return batch
        except Exception as e:
            logging.error(f"Error creating batch: {str(e)}")
            raise

    async def get_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """Checks the status of a specific batch job."""
        try:
            batch = await self.client.batches.retrieve(batch_id)
            return {
                "id": batch.id,
                "status": batch.status,
                "progress": batch.request_counts,
                "created_at": batch.created_at,
                "output_file_id": batch.output_file_id,
                "error_file_id": batch.error_file_id
            }
        except Exception as e:
            logging.error(f"Error checking batch status: {str(e)}")
            raise

    async def get_batch_results(self, batch_id: str) -> Dict[str, Any]:
        """
        Retrieves the results of a completed batch.
        Returns both the output and any errors that occurred.
        """
        try:
            # Get batch status first
            batch = await self.client.batches.retrieve(batch_id)
            
            if batch.status != "completed":
                return {
                    "status": batch.status,
                    "message": f"Batch is not completed yet. Current status: {batch.status}"
                }
            
            results = {}
            
            # Get output file if available
            if batch.output_file_id:
                output_response = await self.client.files.content(batch.output_file_id)
                output_content = output_response.text
                results["output"] = [json.loads(line) for line in output_content.strip().split("\n")]
            
            # Get error file if available
            if batch.error_file_id:
                error_response = await self.client.files.content(batch.error_file_id)
                error_content = error_response.text
                results["errors"] = [json.loads(line) for line in error_content.strip().split("\n")]
            
            return results
        except Exception as e:
            logging.error(f"Error retrieving batch results: {str(e)}")
            raise

    async def cancel_batch(self, batch_id: str) -> Dict[str, Any]:
        """Cancels an ongoing batch job."""
        try:
            batch = await self.client.batches.cancel(batch_id)
            return {
                "id": batch.id,
                "status": batch.status,
                "message": "Batch cancellation initiated"
            }
        except Exception as e:
            logging.error(f"Error cancelling batch: {str(e)}")
            raise

    async def list_batches(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Lists all batch jobs."""
        try:
            batches = await self.client.batches.list(limit=limit)
            batch_list = []
            
            for batch in batches:
                batch_list.append({
                    "id": batch.id,
                    "status": batch.status, 
                    "created_at": batch.created_at,
                    "completed_at": batch.completed_at,
                    "endpoint": batch.endpoint,
                    "metadata": batch.metadata
                })
                
            return batch_list
        except Exception as e:
            logging.error(f"Error listing batches: {str(e)}")
            raise