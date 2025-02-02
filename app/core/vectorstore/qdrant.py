import logging
from qdrant_client import QdrantClient, models
from typing import List, Tuple
from app.core.models.huggingface.huggingface import HuggingFaceEmbeddingGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QdrantDenseSearch:
    def __init__(self, qdrant_url: str, user_id: str, model_name: str):
        self.qdrant_client = QdrantClient(url=qdrant_url)
        self.model_name = model_name
        self.user_id = user_id
        self.collection_name = self._generate_collection_name()
        self.embedding_generator = HuggingFaceEmbeddingGenerator(model_name)

    def _generate_collection_name(self) -> str:
        """Generate a unique collection name based on user_id and model_name"""
        # Convert model name to a valid collection name by replacing invalid characters
        safe_model_name = self.model_name.replace('/', '_').replace('-', '_')
        return f"user_{self.user_id}_{safe_model_name}"

    def _get_embedding_dimension(self, text: str = "Sample text for dimension check") -> int:
        """Get embedding dimension from a sample text"""
        embedding = self.generate_embedding(text)
        return len(embedding)

    def create_collection(self, vector_size: int):
        """Create collection with dynamic vector size"""
        try:
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            logger.info(f"Collection '{self.collection_name}' already exists")
        except Exception:
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE
                )
            )
            logger.info(f"Collection '{self.collection_name}' created with vector size {vector_size}")

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embeddings using HuggingFaceEmbeddingGenerator"""
        return self.embedding_generator.generate_embedding(text)

    def index_documents(self, documents: List[Tuple[int, str]]):
        """Index documents with dynamic collection creation"""
        try:
            # Generate first embedding to get dimension
            if documents:
                first_embedding = self.generate_embedding(documents[0][1])
                vector_size = len(first_embedding)
                
                # Create collection if it doesn't exist
                if not self.collection_exists():
                    self.create_collection(vector_size)

                points = [
                    models.PointStruct(
                        id=doc_id,
                        vector=first_embedding if idx == 0 else self.generate_embedding(text),
                        payload={"text": text}
                    )
                    for idx, (doc_id, text) in enumerate(documents)
                ]
                
                self.qdrant_client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                logger.info(f"Indexed {len(documents)} documents")
            else:
                logger.warning("No documents provided for indexing")

        except Exception as e:
            logger.error(f"Indexing failed: {str(e)}")
            raise

    def search(self, query: str, top_k: int = 5):
        """Perform search with dynamic collection creation"""
        try:
            # Generate query embedding and get dimension
            query_embedding = self.generate_embedding(query)
            vector_size = len(query_embedding)
            
            # Create collection if it doesn't exist
            if not self.collection_exists():
                self.create_collection(vector_size)
                logger.warning("Empty collection - no results will be returned")
                return []
                
            return self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                with_payload=True
            )
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            raise

    def collection_exists(self) -> bool:
        """Check if the collection exists"""
        try:
            self.qdrant_client.get_collection(self.collection_name)
            return True
        except Exception:
            return False

if __name__ == "__main__":
    # Example usage with user-specific collection
    qdrant_search = QdrantDenseSearch(
        qdrant_url="http://localhost:6333",
        user_id="user123",
        model_name="BAAI/bge-large-en-v1.5"
    )

    # Index documents
    documents = [
        (1, "The impact of AI on healthcare is transformative."),
        (2, "Climate change has significant effects on biodiversity."),
        (3, "Renewable energy sources are becoming more cost-effective."),
    ]
    qdrant_search.index_documents(documents)

    # Perform search
    query = "How Tree will effect weather"
    results = qdrant_search.search(query)
    
    print(f"Results for query: '{query}'")
    for result in results:
        print(f"ID: {result.id} | Score: {result.score:.4f} | Text: {result.payload['text']}")
