import logging
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
from app.core.HuggingFace.huggingface import HuggingFaceEmbeddingGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QdrantDenseSearch:
    def __init__(self, qdrant_url: str, collection_name: str):
        self.qdrant_client = QdrantClient(
            url=qdrant_url,
            timeout=600
        )
        self.collection_name = collection_name
        self.embedding_generator = HuggingFaceEmbeddingGenerator('BAAI/bge-large-en-v1.5')

    def create_collection(self):
        if not self.qdrant_client.get_collection(self.collection_name):
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=1024,  # Dimension of BAAI/bge-large-en-v1.5 embeddings
                    distance=models.Distance.COSINE
                )
            )
            logger.info(f"Collection '{self.collection_name}' created successfully")
        else:
            logger.info(f"Collection '{self.collection_name}' already exists")

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embeddings using HuggingFaceEmbeddingGenerator"""
        return self.embedding_generator.generate_embedding(text)

    def index_documents(self, documents: List[Tuple[int, str]]):
        """Index documents with their embeddings"""
        points = [
            models.PointStruct(
                id=doc_id,
                vector=self.generate_embedding(text),
                payload={"text": text}
            )
            for doc_id, text in documents
        ]
        
        self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        logger.info(f"Indexed {len(documents)} documents")

    def search(self, query: str, top_k: int = 5):
        """Perform dense vector search with proper error handling"""
        try:
            query_embedding = self.generate_embedding(query)
            
            return self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,  # Single vector input
                limit=top_k,
                with_payload=True
            )
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            raise

if __name__ == "__main__":
    # Initialize client
    qdrant_search = QdrantDenseSearch(
        qdrant_url="http://localhost:6333",
        collection_name="dense_search"
    )

    # Create collection
    qdrant_search.create_collection()

    # Index documents
    documents = [
        (1, "The impact of AI on healthcare is transformative."),
        (2, "Climate change has significant effects on biodiversity."),
        (3, "Renewable energy sources are becoming more cost-effective."),
    ]
    qdrant_search.index_documents(documents)

    # Perform search
    query = "How is artificial intelligence changing medical care?"
    results = qdrant_search.search(query)
    
    print(f"Results for query: '{query}'")
    for result in results:
        print(f"ID: {result.id} | Score: {result.score:.4f} | Text: {result.payload['text']}")
