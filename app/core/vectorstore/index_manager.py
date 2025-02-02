import numpy as np
from app.core.vectorstore.qdrant_client import QdrantDB
from app.core.models.huggingface.huggingface import HuggingFaceEmbeddingGenerator

class IndexManager:
    """Handles document indexing and embedding generation."""

    def __init__(self, model_name="BAAI/bge-large-en-v1.5"):
        self.qdrant = QdrantDB()
        self.embedding_generator = HuggingFaceEmbeddingGenerator(model_name)

    def generate_embedding(self, text: str):
        """Generates an embedding for a given text chunk."""
        return self.embedding_generator.generate_embedding(text)

    def index_document(self, collection_name: str, text_chunks):
        """Indexes text chunks into Qdrant."""
        embeddings = [self.generate_embedding(chunk) for chunk in text_chunks]
        documents = [{"vector": emb, "text": text, "metadata": {}} for emb, text in zip(embeddings, text_chunks)]
        
        self.qdrant.create_collection(collection_name, vector_size=len(embeddings[0]))
        self.qdrant.insert_vectors(collection_name, documents)