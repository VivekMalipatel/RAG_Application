import uuid
from app.core.vectorstore.qdrant_client import QdrantDB
from app.core.retrieval.cross_modal_linker import CrossModalLinker
from app.core.models.huggingface.huggingface import HuggingFaceEmbeddingGenerator
from app.core.multimodal.embedding_manager import EmbeddingManager

class HybridRetrieval:
    """Handles hybrid search with text and images."""

    def __init__(self):
        self.qdrant = QdrantDB()
        self.text_embedder = HuggingFaceEmbeddingGenerator("BAAI/bge-large-en-v1.5")
        self.embedding_manager = EmbeddingManager()

    def index_multimodal_data(self, text_data, image_data):
        """Indexes multi-modal data into separate Qdrant collections."""
        text_points = []
        image_points = []

        for text in text_data:
            text_embedding = self.text_embedder.generate_embedding(text)
            text_points.append({
                "id": str(uuid.uuid4()),
                "vector": text_embedding,
                "payload": {"type": "text", "content": text}
            })

        for image_path in image_data:
            image_embedding = self.embedding_manager.generate_image_embedding(image_path)
            image_points.append({
                "id": str(uuid.uuid4()),
                "vector": image_embedding,
                "payload": {"type": "image", "path": image_path}
            })

        if text_points:
            self.qdrant.index_vectors(text_points, self.qdrant.text_collection)

        if image_points:
            self.qdrant.index_vectors(image_points, self.qdrant.image_collection)

        print(f"✅ Indexed {len(text_points)} text points and {len(image_points)} image points.")

    def search(self, query, mode="text", top_k=5):
        """Search across text or images."""
        if mode == "text":
            query_vector = self.text_embedder.generate_embedding(query)
            results = self.qdrant.search(query_vector, self.qdrant.text_collection, top_k)
        elif mode == "image":
            query_vector = self.embedding_manager.generate_image_embedding(query)
            results = self.qdrant.search(query_vector, self.qdrant.image_collection, top_k)
        else:
            raise ValueError("Invalid mode. Use 'text' or 'image'.")

        return results