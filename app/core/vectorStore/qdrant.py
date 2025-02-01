import logging
from qdrant_client import QdrantClient, models
from fastembed import SparseTextEmbedding
from typing import List, Tuple, Dict
import numpy as np
from app.core.HuggingFace.huggingface import HuggingFaceEmbeddingGenerator
from qdrant_client.models import SparseVector  # Add this import

class QdrantVectorStore:
    def __init__(self, qdrant_url: str, base_collection_name: str):
        self.qdrant_client = QdrantClient(url=qdrant_url, timeout=600)
        self.base_collection_name = base_collection_name
        self.sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25") 
        self.dense_models = {} 
        self.late_models = {} 
        logging.basicConfig(level=logging.DEBUG)

    def get_collection_name(self, user_id: str, model_dense: str, model_late_interaction: str):
        user_id = user_id.replace('/', '_')
        model_dense = model_dense.replace('/', '_')
        model_late_interaction = model_late_interaction.replace('/', '_')
        return f"{self.base_collection_name}_{user_id}_{model_dense}_{model_late_interaction}"

    def ensure_collection(self, user_id: str, model_dense: str, model_late_interaction: str, dense_vector_size: int, late_vector_size: int):
        collection_name = self.get_collection_name(user_id, model_dense, model_late_interaction)
        logging.debug(f"Ensuring collection: {collection_name}")
        existing_collections = self.qdrant_client.get_collections()
        logging.debug(f"Existing collections: {[col.name for col in existing_collections.collections]}")
        
        if collection_name not in [col.name for col in existing_collections.collections]:
            logging.debug(f"Creating collection: {collection_name}")
            self.qdrant_client.create_collection(
                collection_name,
                vectors_config={
                    model_dense.replace('/', '_'): models.VectorParams(
                        size=dense_vector_size, 
                        distance=models.Distance.COSINE
                    ),
                    model_late_interaction.replace('/', '_'): models.VectorParams(
                        size=late_vector_size, 
                        distance=models.Distance.COSINE,
                        multivector_config=models.MultiVectorConfig(
                            comparator=models.MultiVectorComparator.MAX_SIM,
                        ),
                        hnsw_config=models.HnswConfigDiff(
                            m=0,  # Disable HNSW graph creation
                        ),
                    )
                },
                sparse_vectors_config={
                    "bm25": models.SparseVectorParams(
                        modifier=models.Modifier.IDF
                    )
                }
            )

    def generate_sparse_embedding(self, text: str) -> List[float]:
        """Generate sparse vector using Qdrant BM25 model from Hugging Face."""
        sparse_vec = self.sparse_model.embed(text)
        return list(sparse_vec)[0]

    def generate_dense_embedding(self, text: str, model_dense: str) -> List[float]:
        """Generate dense embeddings using switchable models."""
        if model_dense not in self.dense_models:
            self.dense_models[model_dense] = HuggingFaceEmbeddingGenerator(model_dense)
        return self.dense_models[model_dense].generate_embedding(text)

    def generate_late_interaction_embedding(self, text: str, model_late_interaction: str) -> List[float]:
        """Generate late interaction embeddings."""
        if model_late_interaction not in self.late_models:
            self.late_models[model_late_interaction] = HuggingFaceEmbeddingGenerator(model_late_interaction)
        return self.late_models[model_late_interaction].generate_embedding(text)

    def generate_embeddings(self, text: str, model_dense: str, model_late_interaction: str):
        """Generate all embeddings (dense, sparse, late)."""
        dense_vec = self.generate_dense_embedding(text, model_dense)
        sparse_vec = self.generate_sparse_embedding(text)
        late_vec = self.generate_late_interaction_embedding(text,model_late_interaction)
        return dense_vec, sparse_vec, late_vec


    def store_document(self, user_id: str, model_dense: str, model_late_interaction: str, doc_id: int, text: str):
        """Store document with hybrid embeddings in Qdrant."""
        collection_name = self.get_collection_name(user_id, model_dense, model_late_interaction)
        dense_vec, sparse_vec, late_vec = self.generate_embeddings(text, model_dense, model_late_interaction)
        self.ensure_collection(user_id, model_dense, model_late_interaction, len(dense_vec), len(late_vec))

        print(dense_vec)
        print(sparse_vec)
        print(late_vec)

        self.qdrant_client.upload_points(
            collection_name,
            points=[
                models.PointStruct(
                    id=doc_id,
                    vector={
                        model_dense.replace('/', '_'): dense_vec,
                        "bm25": sparse_vec,
                        model_late_interaction.replace('/', '_'): late_vec,
                    },
                    payload={
                        "text": text,
                    }
                )
            ],
        )

    def hybrid_search(self, user_id: str, model_dense: str, model_late_interaction: str, query: str, top_k: int):
        collection_name = self.get_collection_name(user_id, model_dense, model_late_interaction)
        dense_vec, sparse_vec, late_vec = self.generate_embeddings(query, model_dense, model_late_interaction)
        self.ensure_collection(user_id, model_dense, model_late_interaction, len(dense_vec), len(late_vec))

        prefetch = [
            models.Prefetch(query=dense_vec, using=model_dense.replace('/', '_'), limit=top_k),
            models.Prefetch(query=sparse_vec["bm25"], using="bm25", limit=top_k),
            models.Prefetch(query=late_vec, using=model_late_interaction.replace('/', '_'), limit=top_k)
        ]
        
        results = self.qdrant_client.query_points(
            collection_name,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k
        )
        return results.points
