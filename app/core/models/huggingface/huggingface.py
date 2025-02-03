import logging
import torch
from transformers import AutoTokenizer, AutoModel

class HuggingFaceEmbeddingGenerator:
    """Generates text embeddings using Hugging Face transformer models."""

    def __init__(self, model_name: str):
        """
        Initializes the embedding model.

        Args:
            model_name (str): The name of the Hugging Face model to use.
        """
        self.tokenizer = None
        self.model = None

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            logging.info(f"Successfully loaded Hugging Face model: {model_name}")
        except Exception as e:
            logging.error(f"Error loading Hugging Face model '{model_name}': {e}")

    def generate_embedding(self, text: str) -> list[float]:
        """
        Generates a normalized text embedding.

        Args:
            text (str): Input text.

        Returns:
            list[float]: Normalized embedding vector, or None if an error occurs.
        """
        if not self.model or not self.tokenizer:
            logging.error("Embedding model/tokenizer is not initialized.")
            return None

        try:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Extract CLS token representation
            sentence_embeddings = outputs.last_hidden_state[:, 0]

            # Normalize embeddings
            normalized = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
            
            return normalized.squeeze().tolist()
        except Exception as e:
            logging.error(f"Error generating text embedding: {e}")
            return None

    def check_embedding_size(self, text: str):
        """
        Prints the embedding size for debugging purposes.

        Args:
            text (str): Input text.
        """
        embedding = self.generate_embedding(text)
        if embedding:
            logging.info(f"🔹 Text embedding size: {len(embedding)}")