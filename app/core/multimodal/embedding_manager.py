import torch
import logging
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from app.core.models.huggingface.huggingface import HuggingFaceEmbeddingGenerator

class EmbeddingManager:
    """Generates embeddings for text and images using Hugging Face & CLIP models."""

    def __init__(self, text_model="BAAI/bge-large-en-v1.5", image_model="openai/clip-vit-base-patch32"):
        """
        Initializes embedding models.

        Args:
            text_model (str, optional): Model name for text embedding. Defaults to "BAAI/bge-large-en-v1.5".
            image_model (str, optional): Model name for image embedding. Defaults to "openai/clip-vit-base-patch32".
        """
        try:
            self.text_embedder = HuggingFaceEmbeddingGenerator(text_model)
            self.image_model = CLIPModel.from_pretrained(image_model)
            self.image_processor = CLIPProcessor.from_pretrained(image_model)
            logging.info("Successfully loaded text and image embedding models.")
        except Exception as e:
            logging.error(f"Error initializing embedding models: {e}")
            self.text_embedder = None
            self.image_model = None
            self.image_processor = None

    def generate_text_embedding(self, text: str):
        """
        Generates embeddings for input text.

        Args:
            text (str): Input text.

        Returns:
            list: Text embedding vector or None on failure.
        """
        if not self.text_embedder:
            logging.error("Text embedding model not initialized.")
            return None

        try:
            return self.text_embedder.generate_embedding(text)
        except Exception as e:
            logging.error(f"Error generating text embedding: {e}")
            return None

    def generate_image_embedding(self, image_path: str):
        """
        Generates embeddings for an input image.

        Args:
            image_path (str): Path to the image file.

        Returns:
            list: Image embedding vector or None on failure.
        """
        if not self.image_model or not self.image_processor:
            logging.error("Image embedding model not initialized.")
            return None

        try:
            image = Image.open(image_path)
            inputs = self.image_processor(images=image, return_tensors="pt")

            with torch.no_grad():
                outputs = self.image_model.get_image_features(**inputs)

            return outputs.squeeze().tolist()
        except FileNotFoundError:
            logging.error(f"Image file not found: {image_path}")
        except Exception as e:
            logging.error(f"Error generating image embedding: {e}")

        return None

    def check_embedding_size(self, text: str):
        """
        Prints the embedding size for debugging purposes.

        Args:
            text (str): Input text.
        """
        embedding = self.generate_text_embedding(text)
        if embedding:
            logging.info(f"Text embedding size: {len(embedding)}")