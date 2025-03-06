import logging
from typing import List, Dict, Any, Optional
from qdrant_client.http.models import (
     VectorParams, PointStruct, Filter, Distance, SparseVectorParams, SparseIndexParams, OptimizersConfigDiff, Prefetch, Filter, SparseVector, models
)
from app.core.vector_store.qdrant.qdrant_session import qdrant_session
from app.core.models.model_handler import ModelRouter
from app.core.models.model_type import ModelType
from app.core.models.model_provider import Provider
import uuid
import numpy as np

class QdrantHandler:
    """Handles vector operations with Qdrant for hybrid search with dense and sparse vectors."""

    def __init__(self):
        self.reranker = ModelRouter(
            provider=Provider.HUGGINGFACE,
            model_name="jinaai/jina-colbert-v2",
            model_type=ModelType.RERANKER
        )

    async def create_collection(
        self,
        user_id: str,
        dense_vector_size: int = 768,
        matryoshka_sizes: list = [64, 128, 256],
        quantized_size: int = 768,
        sparse_enabled: bool = True
    ):
        """Creates a user-specific collection in Qdrant for hybrid search."""
        try:
            if not user_id:
                raise ValueError("user_id cannot be empty")

            # Define vector configurations
            vectors_config = {
                "dense": VectorParams(
                    size=dense_vector_size,
                    distance=Distance.COSINE,
                    on_disk=True
                ),
                "quantized": VectorParams(
                    size=quantized_size,
                    distance=Distance.COSINE,
                    on_disk=True
                )
            }

            # Add Matryoshka embeddings
            for dim in matryoshka_sizes:
                vectors_config[f"matryoshka_{dim}"] = VectorParams(
                    size=dim,
                    distance=Distance.COSINE,
                    on_disk=False
                )

            # Add Sparse embeddings (BM25 for text, TBD for image)
            sparse_vectors_config = None
            if sparse_enabled:
                sparse_vectors_config = {
                    "sparse": SparseVectorParams(
                        index=SparseIndexParams(on_disk=False)
                    )
                }

            # Optimize memory & indexing
            optimizers_config = OptimizersConfigDiff(
                memmap_threshold=20000
            )

            # Create collection
            await qdrant_session.client.recreate_collection(
                collection_name=user_id,
                vectors_config=vectors_config,
                sparse_vectors_config=sparse_vectors_config,
                optimizers_config=optimizers_config,
                on_disk_payload=True
            )

            logging.info(f"Created hybrid search collection for user {user_id}")

        except ValueError as ve:
            logging.error(f"Validation error creating collection for user {user_id}: {str(ve)}")
            raise
        except Exception as e:
            logging.error(f"Failed to create collection for user {user_id}: {str(e)}")
            raise


    async def store_vectors(self, embedded_chunks: List[Dict[str, Any]], user_id: str):
        """
        Stores document chunks with multi-stage embeddings (dense, sparse, matryoshka, image).

        Args:
            embedded_chunks (list): List of chunks with various embeddings.
            metadata (dict): Document metadata including user_id.
        """
        try:
            # Check if collection exists, if not create it
            collections = await qdrant_session.client.get_collections()
            if str(user_id) not in [c.name for c in collections.collections]:
                await self.create_collection(user_id=user_id)

            points = []

            for chunk in embedded_chunks:
                # Validate Dense Embedding Dimension
                if len(chunk["dense_embedding"]) != 768:
                    raise ValueError(f"Dense vector dimension mismatch. Expected 768, got {len(chunk['dense_embedding'])}")

                # Generate a unique identifier for the chunk
                chunk_id = str(uuid.uuid4())

                quantized_dense_embedding = np.clip(
                    (np.array(chunk["dense_embedding"]) * 127).astype(np.int8), -128, 127
                ).tolist()
                dense_embedding = np.array(chunk["dense_embedding"])
                matryoshka_64 = dense_embedding[:64].tolist()
                matryoshka_128 = dense_embedding[:128].tolist()
                matryoshka_256 = dense_embedding[:256].tolist()

                metadata = chunk["chunk_metadata"]

                # Define the point structure with all embeddings
                point = PointStruct(
                    id=chunk_id,
                    vector={
                        "dense": chunk["dense_embedding"],
                        "matryoshka_64": matryoshka_64,
                        "matryoshka_128": matryoshka_128,
                        "matryoshka_256": matryoshka_256,
                        "quantized": quantized_dense_embedding,
                        "sparse": chunk["sparse_embedding"]
                    },
                    payload={
                        "document_id": metadata["document_id"],
                        "user_id": metadata["user_id"],
                        "file_name": metadata["file_name"],
                        "mime_type": metadata["mime_type"],
                        "file_size": metadata["file_size"],
                        "file_description": metadata["description"],
                        "file_path": metadata["file_path"],
                        "context_version": metadata["context_version"],
                        "chunk_number": metadata['chunk_number'],
                        "entities": metadata.get("entities"),
                        "context": metadata.get("context"), 
                        "document_summary": metadata["doc_summary"],
                        "content": str(chunk["content"]),
                        "page_number": metadata.get("page_number"),
                        "languages": metadata.get("languages"),
                        "element_id": metadata.get("element_id"),
                        "is_continuation": metadata.get("is_continuation"),
                        "category": metadata.get("category"),
                    }
                )
                points.append(point)

            # Upsert the processed chunks into Qdrant
            await qdrant_session.client.upsert(
                collection_name=user_id,
                points=points
            )
            logging.info(f"Stored {len(points)} chunks with multi-stage embeddings for user {user_id}")

        except Exception as e:
            logging.error(f"Failed to store vectors: {str(e)}")
            raise

    async def hybrid_search(
        self,
        user_id: str,
        query_text: str,
        dense_vector: List[float],
        sparse_vector: Dict[str, List[float]],
        image_embedding: Optional[List[float]] = None,
        top_k: int = 10,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Performs multi-stage hybrid search using Matryoshka embeddings, quantized dense search, sparse search, and late interaction reranking.

        Args:
            user_id (str): Collection name.
            query_text (str): Original text query.
            dense_vector (List[float]): Full 768D dense embedding of the query.
            sparse_vector (Dict[str, List[float]]): Sparse vector representation (BM25 for text).
            image_embedding (Optional[List[float]]): Optional image embedding for multimodal search.
            top_k (int): Number of results to return.
            filters (Optional[Dict]): Additional search filters.

        Returns:
            List[Dict]: Final ranked results.
        """
        try:
            query_filter = Filter(**filters) if filters else None

            quantized_query = np.clip(
                (np.array(dense_vector) * 127).astype(np.int8), -128, 127
            ).tolist()

            # **Step A: Matryoshka Dimensionality Reduction Search**
            matryoshka_prefetch = Prefetch(
                            prefetch=[
                                Prefetch(
                                    prefetch=[
                                        Prefetch(
                                            prefetch=[
                                                # First stage - 64D retrieval (high recall, low precision)
                                                Prefetch(
                                                    query=dense_vector[:64],
                                                    using="matryoshka_64",
                                                    limit=400,
                                                ),
                                            ],
                                            # Second stage - 128D refinement
                                            query=dense_vector[:128],
                                            using="matryoshka_128",
                                            limit=300,
                                        ),
                                    ],
                                    # Third stage - 256D final refinement
                                    query=dense_vector[:256],
                                    using="matryoshka_256",
                                    limit=200,
                                ),
                            ],
                            # Fourth stage - Full 768D precision refinement
                            query=dense_vector,
                            using="dense",
                            limit=10,
                        )

            # **Step B: Integer-Based Quantized Search & 768D Dense Re-ranking**
            quantized_dense_prefetch = Prefetch(
                prefetch=[
                    # Integer-based search for speed
                    Prefetch(
                        query=quantized_query,
                        using="quantized",
                        limit=30,
                    )
                ],
                # Refine with full precision 768D float vectors
                query=dense_vector,
                using="dense",
                limit=10,
            )

            # **Step C: Sparse Search (BM25 for text, TBD for images)**
            sparse_search_prefetch = Prefetch(
                query=sparse_vector if isinstance(sparse_vector, SparseVector) else SparseVector(
                    indices=sparse_vector["indices"],
                    values=sparse_vector["values"]
                ),
                using="sparse",
                limit=10,
            )

            # **Step D: Fusion of Quantized Dense + Sparse using RRF**
            sparse_dense_rrf = Prefetch(
                prefetch=[quantized_dense_prefetch, sparse_search_prefetch],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
            )

            # **Execute Hybrid Search Pipeline**
            search_result = await qdrant_session.client.query_points(
                collection_name=user_id,
                prefetch=[matryoshka_prefetch, sparse_dense_rrf],
                query=dense_vector,
                using="dense",
                limit=10,
                search_params={"hnsw_ef": 128, "exact": True},
                with_payload=True,
                query_filter=query_filter
            )

            results = search_result.points

            # **Extract Documents for Reranking**
            documents = [res.payload["content"] for res in results if res.payload and "content" in res.payload]

            if not documents:
                logging.warning("No documents retrieved for reranking.")
                return results  # Return original results if no content found

            # **Step E: Apply Late Interaction Reranking (ColBERT-V2)**
            reranked_results = await self.rerank_with_colbert(query_text, documents, results)

            return reranked_results[:top_k]

        except Exception as e:
            logging.error(f"Hybrid search failed for user {user_id}: {str(e)}")
            return []

    async def rerank_with_colbert(self, query: str, documents: List[str], results: List[Dict]) -> List[Dict]:
        """
        Uses Hugging Face's JinaAI ColBERT V2 for reranking.

        Args:
            query (str): User query.
            documents (List[str]): List of retrieved documents.
            results (List[Dict]): Initial hybrid search results.

        Returns:
            List[Dict]: Reranked search results.
        """
        try:
            ranked_indices = self.reranker.client.rerank_documents(query, documents)

            if not ranked_indices:
                return results

            # Reorder search results based on reranked indices
            reranked_results = [results[i] for i in ranked_indices]

            return reranked_results
        except Exception as e:
            logging.error(f"Reranking failed: {str(e)}")
            return results