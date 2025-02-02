from transformers import CLIPProcessor, CLIPModel
import torch
from app.core.models.huggingface.huggingface import HuggingFaceEmbeddingGenerator

class EmbeddingManager:
    """Generates embeddings for text, images, and audio."""

    def __init__(self, text_model="BAAI/bge-large-en-v1.5", image_model="openai/clip-vit-base-patch32"):
        self.text_embedder = HuggingFaceEmbeddingGenerator(text_model)
        self.image_model = CLIPModel.from_pretrained(image_model)
        self.image_processor = CLIPProcessor.from_pretrained(image_model)

    def generate_text_embedding(self, text: str):
        """Generates text embeddings."""
        return self.text_embedder.generate_embedding(text)

    def generate_image_embedding(self, image_path: str):
        """Generates image embeddings."""
        from PIL import Image
        image = Image.open(image_path)
        inputs = self.image_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.image_model.get_image_features(**inputs)
        return outputs.squeeze().tolist()
    
    def check_embedding_size(model, text):
        embedding = model.generate_embedding(text)
        print(f"Embedding size: {len(embedding)}")