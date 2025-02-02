from qdrant_client import QdrantClient, models
import os
import logging
from dotenv import load_dotenv
from app.core.models.huggingface.huggingface import HuggingFaceEmbeddingGenerator
from app.core.multimodal.embedding_manager import EmbeddingManager

load_dotenv()

class QdrantDB:
    """Manages Qdrant collections and indexing for multi-modal embeddings."""

    def __init__(self, text_collection="text_collection", image_collection="image_collection"):
        self.client = QdrantClient(url=f"http://{os.getenv('QDRANT_HOST', 'localhost')}:6333")
        self.text_collection = text_collection
        self.image_collection = image_collection
        self.embedding_manager = EmbeddingManager()
        self.text_vector_size, self.image_vector_size = self.detect_vector_sizes()
        self.ensure_collections_exist()

    def detect_vector_sizes(self):
        """Detect vector sizes for text and image embeddings."""
        text_embedding = HuggingFaceEmbeddingGenerator("BAAI/bge-large-en-v1.5").generate_embedding("Sample text")
        image_embedding = self.embedding_manager.generate_image_embedding("Temp/image/image.png")

        text_size = len(text_embedding)
        image_size = len(image_embedding)

        print(f"📌 Detected vector sizes -> Text: {text_size}, Image: {image_size}")
        return text_size, image_size

    def create_collection(self, collection_name: str, vector_size: int):
        """Creates a Qdrant collection if it does not exist."""
        if not self.client.collection_exists(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
            )
            print(f"✅ Collection '{collection_name}' created with vector size {vector_size}.")

    def ensure_collections_exist(self):
        """Ensures both collections exist with correct vector sizes."""
        self.create_collection(self.text_collection, self.text_vector_size)
        self.create_collection(self.image_collection, self.image_vector_size)

    def index_vectors(self, points, collection_name):
        """Indexes vectors into the appropriate Qdrant collection."""
        self.client.upsert(collection_name=collection_name, points=points)
        print(f"✅ Indexed {len(points)} documents in '{collection_name}'.")

    def search(self, query_vector, collection_name, top_k=5):
        """Searches the correct collection based on vector dimension."""
        return self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True
        )