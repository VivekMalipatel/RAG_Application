import logging
from typing import Optional, List, Dict, Any, Union, AsyncGenerator
from openai import AsyncOpenAI, APIError
from app.config import settings
import aiofiles


# Default Prompt Template (Can Be Overridden)
DEFAULT_TEMPLATE = """{{ if .System }}<|start_header_id|>system<|end_header_id|>
{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>
{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>
{{ .Response }}<|eot_id|>"""

# Default Stop Sequences
DEFAULT_STOP_SEQUENCES = ["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"]

class OpenAIClient:
    """
    Unified OpenAI client supporting:
    - Text generation (GPT-4, GPT-3.5, DeepSeek, etc.)
    - Embeddings (text-embedding-ada-002, etc.)
    - Image generation (DALL·E)
    - Audio processing (Whisper)
    """

    def __init__(
        self,
        model_name: str = "gpt-4-turbo",
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 1024,
        stream: bool = False,
        functions: Optional[List[Dict[str, Any]]] = None,
        function_call: Optional[str] = None,
        embedding: bool = False,
        template: str = DEFAULT_TEMPLATE,
        stop_sequences: List[str] = DEFAULT_STOP_SEQUENCES,
        image_quality: str = "standard",  # Added image quality param
        image_style: str = "vivid"        # Added image style param
    ):
        """
        Initializes an OpenAI client with customized settings.

        Args:
            model_name (str): Model name (e.g., "gpt-4-turbo", "text-embedding-ada-002").
            system_prompt (Optional[str]): Custom system instructions.
            temperature (float): Controls randomness in generation.
            top_p (float): Probability mass for nucleus sampling.
            max_tokens (int): Maximum response length.
            stream (bool): Enables streaming responses.
            functions (Optional[List[Dict[str, Any]]]): OpenAI function calling.
            function_call (Optional[str]): Name of function to call.
            embedding (bool): Whether to use an embedding model.
            template (str): Customizable prompt template.
            stop_sequences (List[str]): Custom stop sequences.
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
        self.functions = functions
        self.function_call = function_call
        self.embedding = embedding
        self.template = template
        self.stop_sequences = stop_sequences
        self.image_quality = image_quality
        self.image_style = image_style

    async def generate_text(self, user_prompt: str, messages: Optional[List[Dict[str, str]]] = None) -> AsyncGenerator[str, None]:
        """
        Generates a response from OpenAI LLMs (GPT-4, DeepSeek, etc.).
        Uses customizable templates & system prompts.
        """
        if self.embedding:
            raise ValueError("Use `embed_text` for embeddings")

        formatted_prompt = self._apply_template(user_prompt)
        formatted_messages = self._prepare_messages(formatted_prompt, messages)

        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=formatted_messages,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                stream=self.stream,
                functions=self.functions,
                function_call={"name": self.function_call} if self.function_call else None
            )

            if self.stream:
                async for chunk in response:
                    yield chunk.choices[0].delta.content if chunk.choices[0].delta.content else ""

            yield response.choices[0].message.content

        except APIError as e:
            logging.error(f"OpenAI API Error: {str(e)}")
            yield f"API Error: {str(e)}"
        except Exception as e:
            logging.error(f"Unexpected Error: {str(e)}")
            yield "Error processing request"
        finally:
            if not self.stream:
                return
    
    async def generate_image(
        self,
        prompt: str,
        model: str = "dall-e-3",
        size: str = "1024x1024",
        quality: Optional[str] = None,
        style: Optional[str] = None,
        response_format: str = "url"
    ) -> Union[str, bytes]:
        """
        Generates images using DALL·E with enhanced configuration options.
        
        Args:
            prompt: Text description of desired image
            model: Image generation model (dall-e-2/dall-e-3)
            size: Output dimensions (256x256, 512x512, 1024x1024, 1792x1024, 1024x1792)
            quality: "standard" or "hd" (DALL·E 3 only)
            style: "vivid" or "natural" (DALL·E 3 only)
            response_format: "url" or "b64_json"

        Returns:
            URL or base64 encoded image data
        """
        try:
            response = await self.client.images.generate(
                model=model,
                prompt=prompt,
                size=size,
                quality=quality or self.image_quality,
                style=style or self.image_style,
                response_format=response_format
            )

            if response_format == "url":
                return response.data[0].url
            return response.data[0].b64_json

        except APIError as e:
            logging.error(f"Image Generation Error: {str(e)}")
            return f"Image API Error: {str(e)}"
        except Exception as e:
            logging.error(f"Image Generation Failed: {str(e)}")
            return "Error generating image"
        
    async def transcribe_audio(
        self,
        audio_path: str,
        model: str = "whisper-1",
        language: Optional[str] = None,
        temperature: float = 0.2,
        prompt: Optional[str] = None
    ) -> str:
        """
        Transcribes audio files using Whisper with enhanced audio handling.
        
        Args:
            audio_path: Path to audio file (mp3, mp4, mpeg, mpga, m4a, wav, webm)
            model: Whisper model version
            language: ISO-639-1 language code
            temperature: Control randomness (0-1)
            prompt: Optional contextual text

        Returns:
            Transcribed text
        """
        try:
            async with aiofiles.open(audio_path, "rb") as audio_file:
                transcript = await self.client.audio.transcriptions.create(
                    file=audio_file,
                    model=model,
                    language=language,
                    temperature=temperature,
                    prompt=prompt
                )
            return transcript.text

        except FileNotFoundError:
            logging.error(f"Audio file not found: {audio_path}")
            return "Error: Audio file not found"
        except APIError as e:
            logging.error(f"Transcription Error: {str(e)}")
            return f"Audio API Error: {str(e)}"
        except Exception as e:
            logging.error(f"Transcription Failed: {str(e)}")
            return "Error processing audio"

    async def embed_text(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """
        Generates text embeddings using OpenAI.
        Supports both single & batch text inputs.

        Args:
            texts (Union[str, List[str]]): The input text(s) to embed.

        Returns:
            List[List[float]]: A list of embeddings.
        """
        if not self.embedding:
            raise ValueError("Use `generate` for text generation")

        try:
            response = await self.client.embeddings.create(
                model=self.model_name,
                input=[texts] if isinstance(texts, str) else texts
            )
            return [item.embedding for item in response.data]

        except APIError as e:
            logging.error(f"Embedding API Error: {str(e)}")
            return []
        except Exception as e:
            logging.error(f"Embedding Error: {str(e)}")
            return []
        

    async def get_model_list(self) -> List[str]:
        """
        Retrieves available models from OpenAI.
        Supports filtering by model type (text, embedding, image, etc.).

        Returns:
            List[str]: List of available model names.
        """
        try:
            models = await self.client.models.list()
            return [model.id for model in models.data]
        except APIError as e:
            logging.error(f"Model List Error: {str(e)}")
            return []

    def _apply_template(self, user_prompt: str) -> str:
        """
        Applies a structured template to the input prompt.
        Dynamically replaces placeholders with relevant data.
        """
        formatted_prompt = self.template.replace("{{ .System }}", self.system_prompt or "")
        formatted_prompt = formatted_prompt.replace("{{ .Prompt }}", user_prompt)
        formatted_prompt = formatted_prompt.replace("{{ .Response }}", "")  # Initial response is empty
        return formatted_prompt

    def _prepare_messages(self, user_prompt: str, messages: Optional[List[Dict[str, str]]]) -> List[Dict[str, str]]:
        """
        Prepares chat messages with system prompts and user input.
        """
        chat_messages = messages or []
        if self.system_prompt:
            chat_messages.insert(0, {"role": "system", "content": self.system_prompt})
        chat_messages.append({"role": "user", "content": user_prompt})
        return chat_messages