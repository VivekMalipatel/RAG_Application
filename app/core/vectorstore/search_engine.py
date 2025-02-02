from rank_bm25 import BM25Okapi
from app.core.vectorstore.qdrant_client import QdrantDB
import numpy as np

class SearchEngine:
    """Implements hybrid search using BM25 and Qdrant."""

    def __init__(self, collection_name):
        self.qdrant = QdrantDB()
        self.collection_name = collection_name
        self.corpus = []
        self.bm25 = None

    def build_bm25(self, text_chunks):
        """Builds a BM25 index for keyword search."""
        self.corpus = [chunk.split() for chunk in text_chunks]
        self.bm25 = BM25Okapi(self.corpus)

    def search(self, query: str, query_vector, top_k=5):
        """Performs a hybrid search (BM25 + Vector)."""
        
        # Compute BM25 scores safely
        if self.bm25:
            bm25_scores = self.bm25.get_scores(query.split())
        else:
            bm25_scores = np.zeros(len(self.corpus))  # Default to zero scores

        vector_results = self.qdrant.search(self.collection_name, query_vector, top_k)

        # Combine BM25 + Vector scores
        hybrid_results = []
        for idx, result in enumerate(vector_results):
            bm25_score = bm25_scores[idx] if idx < len(bm25_scores) else 0  # Avoid index errors
            final_score = bm25_score + result.score
            hybrid_results.append((result.payload["text"], final_score))

        return sorted(hybrid_results, key=lambda x: x[1], reverse=True)