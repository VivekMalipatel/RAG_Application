import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from qdrant_client.http.models import (
     VectorParams, PointStruct, 
    Filter, ScoredPoint, Distance, SparseVectorParams, SparseIndexParams
)
from app.core.vector_store.qdrant.qdrant_session import qdrant_session
import uuid

class QdrantHandler:
    """Handles vector operations with Qdrant for hybrid search with dense and sparse vectors."""

    async def create_collection(
        self,
        user_id: str,
        dense_vector_size: int = 768,
        distance_metric: str = "Cosine"
        ):
        """Creates a user-specific collection in Qdrant for hybrid search."""
        try:
            # Validate inputs
            if not user_id:
                raise ValueError("user_id cannot be empty")
            
            await qdrant_session.client.recreate_collection(
                collection_name=user_id,
                vectors_config={
                    "dense": VectorParams(
                        size=dense_vector_size,
                        distance=Distance[distance_metric.upper()]
                    )
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(
                        index=SparseIndexParams(
                            on_disk=False,
                        )
                    )
                }
            )
            logging.info(f"Created hybrid search collection for user {user_id}")

        except ValueError as ve:
            logging.error(f"Validation error creating collection for user {user_id}: {str(ve)}")
            raise
        except Exception as e:
            logging.error(f"Failed to create collection for user {user_id}: {str(e)}")
            raise


    async def store_vectors(self, embedded_chunks: list, metadata: Dict[str, Any]):
        """
        Stores document chunks with both dense and sparse vectors.

        Args:
            embedded_chunks (list): List of chunks with dense and sparse embeddings
            metadata (dict): Document metadata including user_id
        """
        try:
            
            user_id = metadata["user_id"]

            # Check if collection exists, if not create it
            collections = await qdrant_session.client.get_collections()
            if user_id not in [c.name for c in collections.collections]:
                await self.create_collection(
                    user_id=user_id,
                    dense_vector_size=768
                )

            points = []

            for i, chunk in enumerate(embedded_chunks):

                if len(chunk["dense_embedding"]) != 768:
                    raise ValueError(f"Dense vector dimension mismatch. Expected 768, got {len(chunk['dense_embedding'])}")

                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector={
                        "dense": chunk["dense_embedding"],
                        "sparse": chunk["sparse_embedding"]
                    },
                    payload={
                        "document_id": metadata["document_id"],
                        "user_id": metadata["user_id"],
                        "file_path": metadata["file_path"],
                        "timestamp": metadata["timestamp"],
                        "context_version": metadata["context_version"],
                        "content": str(chunk["content"])
                    }
                )
                points.append(point)

            await qdrant_session.client.upsert(
                collection_name=user_id,
                points=points
            )
            logging.info(f"Stored {len(points)} chunks with hybrid vectors for user {user_id}")

        except Exception as e:
            logging.error(f"Failed to store vectors: {str(e)}")
            raise

    async def hybrid_search(
        self,
        user_id: str,
        dense_vector: List[float],
        sparse_vector: List[float],
        top_k: int = 5,
        filters: Optional[Dict] = None
    ) -> List[ScoredPoint]:
        """
        Performs hybrid search using both dense and sparse vectors.

        Args:
            user_id (str): Collection name
            dense_vector (List[float]): Dense query vector
            sparse_vector (List[float]): Sparse query vector
            top_k (int): Number of results to return
            filters (Optional[Dict]): Additional search filters

        Returns:
            List[ScoredPoint]: Ranked search results
        """
        try:
            query_filter = Filter(**filters) if filters else None
            
            results = await qdrant_session.client.search(
                collection_name=user_id,
                query_vector={
                    "dense": dense_vector,
                    "sparse": sparse_vector
                },
                query_filter=query_filter,
                limit=top_k,
                search_params={
                    "hnsw_ef": 128,
                    "exact": False
                }
            )
            return results

        except Exception as e:
            logging.error(f"Hybrid search failed for user {user_id}: {str(e)}")
            return []

    async def delete_user_collection(self, user_id: str):
        """Deletes a user's vector collection."""
        try:
            await qdrant_session.client.delete_collection(user_id)
            logging.info(f"Deleted collection for user {user_id}")
        except Exception as e:
            logging.error(f"Failed to delete collection for user {user_id}: {str(e)}")
            raise