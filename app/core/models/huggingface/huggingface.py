import torch
import logging
from transformers import AutoTokenizer, AutoModel, AutoProcessor, pipeline
from PIL import Image
from typing import List, Union, Optional, Dict
import requests
from app.config import settings
import numpy as np

# Default Prompt Template (Can Be Overridden)
DEFAULT_TEMPLATE = """{{ if .System }}<|start_header_id|>system<|end_header_id|>
{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>
{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>
{{ .Response }}<|eot_id|>"""

# Default Stop Sequences
DEFAULT_STOP_SEQUENCES = ["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"]

class HuggingFaceClient:
    """
    Unified client for interacting with Hugging Face models.
    Supports:
    - Text embeddings (Nomic, BGE, etc.)
    - Vision embeddings (CLIP, Nomic, etc.)
    - Text generation (LLaMA, Mistral, etc.)
    - Image generation (Stable Diffusion, SDXL, etc.)
    - **Document reranking using Jina ColBERT V2**
    """

    def __init__(
        self,
        model_name: str,
        model_type: str = "text",
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 1024,
        stream: bool = False
    ):
        """
        Initializes a Hugging Face model client.

        Args:
            model_name (str): Hugging Face model repository ID.
            model_type (str): One of ['text', 'image', 'generation', 'image-generation'].
            system_prompt (Optional[str]): Custom system prompt for the model.
            temperature (float): Sampling temperature for randomness.
            top_p (float): Probability mass for nucleus sampling.
            max_tokens (int): Maximum tokens allowed in response.
            stream (bool): Whether to return streaming output.
            stop_sequences (Optional[List[str]]): Custom stop sequences.
            device (str, optional): Device to run inference ('cuda' or 'cpu'). Defaults to best available.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.model_type = model_type
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.stream = stream
        self.stop_sequences = DEFAULT_STOP_SEQUENCES
        self.template = DEFAULT_TEMPLATE
        self.model = None
        self.tokenizer = None
        self.processor = None

        try:
            # Add model version pinning and trust remote code
            kwargs = {
                "trust_remote_code": True,
                "revision": "main",  # Pin to specific version if needed
                "device_map": self.device
            }

            if model_type in ["text", "generation", "reranker"]:
                self._load_text_model()
            elif model_type == "image":
                self._load_image_model(**kwargs)
            elif model_type == "image-generation":
                self._load_image_generation_model()
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
        except Exception as e:
            logging.error(f"Failed to load Hugging Face model {model_name}: {str(e)}")
            raise RuntimeError(f"Model initialization failed: {model_name}")

    def _load_text_model(self):
        """Loads a text model for embeddings, generation, or reranking."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)
            self.model.eval()
            self.model.to(self.device)
            self.max_length = min(8194, self.tokenizer.model_max_length)
            logging.info(f"Loaded text model: {self.model_name} on {self.device}")
        except Exception as e:
            logging.error(f"Text model loading failed: {str(e)}")
            raise

    def _load_image_model(self, **kwargs):
        """Loads an image embedding model (e.g., CLIP, Nomic-vision)."""
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name, **kwargs)
            self.model.eval()
            self.model.to(self.device)
            logging.info(f"Loaded image embedding model: {self.model_name} on {self.device}")
        except Exception as e:
            logging.error(f"Image model loading failed: {str(e)}")
            raise

    def _load_image_generation_model(self, **kwargs):
        """Loads an image generation model (e.g., Stable Diffusion)."""
        self.model = pipeline("text-to-image", model=self.model_name, device=0 if self.device == "cuda" else -1)
        logging.info(f"Loaded image generation model: {self.model_name}")

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """Encodes a list of text inputs into embeddings."""
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=self.max_length ).to(self.device)
        with torch.no_grad():
            embeddings = self.model(**inputs).last_hidden_state.mean(dim=1)
        return embeddings.cpu().numpy()

    def encode_image(self, image_paths: List[str]) -> torch.Tensor:
        """Encodes a list of image file paths into embeddings."""
        images = [Image.open(path) for path in image_paths]
        inputs = self.processor(images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            embeddings = self.model(**inputs).last_hidden_state.mean(dim=1)
        return embeddings.cpu().numpy()

    def generate_text(self, prompt: str, context: Optional[List[Dict[str, str]]] = None, max_tokens: int = None) -> str:
        """Generates text using a Hugging Face text generation model."""
        formatted_prompt = self._apply_template(prompt)

        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        output = self.model.generate(
            **inputs,
            max_length=max_tokens if max_tokens else self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p
        )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def generate_image(self, prompt: str) -> Image:
        """Generates an image using a Hugging Face image generation model."""
        return self.model(prompt)[0]["image"]

    def _apply_template(self, user_prompt: str) -> str:
        """Applies a template to structure the prompt dynamically."""
        formatted_prompt = self.template.replace("{{ .System }}", self.system_prompt or "")
        formatted_prompt = formatted_prompt.replace("{{ .Prompt }}", user_prompt)
        formatted_prompt = formatted_prompt.replace("{{ .Response }}", "")  # Initial response is empty
        return formatted_prompt
    

    def rerank_documents(self, query: str, documents: List[str]) -> List[int]:
        """Uses JinaAI ColBERT V2 for reranking."""
        query_tokens = self.tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=self.max_length).to(self.device)
        doc_tokens = self.tokenizer(documents, return_tensors="pt", truncation=True, padding=True, max_length=self.max_length).to(self.device)

        with torch.no_grad():
            query_embedding = self.model(**query_tokens).last_hidden_state.mean(dim=1)
            doc_embeddings = self.model(**doc_tokens).last_hidden_state.mean(dim=1)
            scores = torch.matmul(query_embedding, doc_embeddings.T).squeeze().cpu().numpy()

        ranked_indices = np.argsort(scores)[::-1]
        return ranked_indices.tolist()