import numpy as np
from app.core.vectorstore.qdrant_client import QdrantDB

class CrossModalLinker:
    """Links text, images, and audio based on embedding similarity."""

    def __init__(self, collection_name="multimodal_collection"):
        self.qdrant = QdrantDB(collection_name)
        self.collection_name = collection_name

    def cosine_similarity(self, vec1, vec2):
        """Calculates cosine similarity between two vectors."""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def link_embeddings(self, text_embeddings, image_embeddings, audio_embeddings, threshold=0.7):
        """Finds links between text, image, and audio based on similarity."""
        links = []

        for text_idx, text_vec in enumerate(text_embeddings):
            for image_idx, image_vec in enumerate(image_embeddings):
                score = self.cosine_similarity(text_vec, image_vec)
                if score > threshold:
                    links.append(("text", text_idx, "image", image_idx, score))

            for audio_idx, audio_vec in enumerate(audio_embeddings):
                score = self.cosine_similarity(text_vec, audio_vec)
                if score > threshold:
                    links.append(("text", text_idx, "audio", audio_idx, score))

        return links