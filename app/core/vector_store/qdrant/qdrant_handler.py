import logging
from typing import List, Dict, Any, Optional
from qdrant_client.http.models import (
     VectorParams, PointStruct, Filter, Distance, SparseVectorParams, SparseIndexParams, OptimizersConfigDiff, FusionQuery, Prefetch, Filter, SparseVector
)
from colbert.modeling.checkpoint import Checkpoint
from colbert.infra import ColBERTConfig
import torch
from app.core.vector_store.qdrant.qdrant_session import qdrant_session
import uuid
import numpy as np

class QdrantHandler:
    """Handles vector operations with Qdrant for hybrid search with dense and sparse vectors."""

    async def create_collection(
        self,
        user_id: str,
        dense_vector_size: int = 768,
        image_vector_size: int = 512,
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
                "image": VectorParams(
                    size=image_vector_size,
                    distance=Distance.DOT,
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
                    "sparse_text": SparseVectorParams(
                        index=SparseIndexParams(on_disk=False)
                    ),
                    "sparse_image": SparseVectorParams(
                        index=SparseIndexParams(on_disk=True)
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


    async def store_vectors(self, embedded_chunks: List[Dict[str, Any]], metadata: Dict[str, Any]):
        """
        Stores document chunks with multi-stage embeddings (dense, sparse, matryoshka, image).

        Args:
            embedded_chunks (list): List of chunks with various embeddings.
            metadata (dict): Document metadata including user_id.
        """
        try:
            user_id = metadata["user_id"]

            # Check if collection exists, if not create it
            collections = await qdrant_session.client.get_collections()
            if user_id not in [c.name for c in collections.collections]:
                await self.create_collection(user_id=user_id)

            points = []

            for i, chunk in enumerate(embedded_chunks):
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

                # Define the point structure with all embeddings
                point = PointStruct(
                    id=chunk_id,
                    vector={
                        "dense": chunk["dense_embedding"],
                        "matryoshka_64": matryoshka_64,
                        "matryoshka_128": matryoshka_128,
                        "matryoshka_256": matryoshka_256,
                        "quantized": quantized_dense_embedding,
                        "image": chunk.get("image_embedding", []),  # Image embedding if available
                        "sparse_text": chunk["sparse_text_embedding"],
                        "sparse_image": chunk.get("sparse_image_embedding", []),  # TBD method
                    },
                    payload={
                        "document_id": metadata["document_id"],
                        "user_id": metadata["user_id"],
                        "file_path": metadata["file_path"],
                        "timestamp": metadata["timestamp"],
                        "context_version": metadata["context_version"],
                        "chunk_number": i,  # Track chunk position
                        "hierarchy": chunk.get("document_hierarchy", []),  # Extracted document structure
                        "entities": chunk.get("entities", []),  # Named entities
                        "context": chunk.get("context", ""),  # Contextual metadata
                        "position": chunk.get("position", ""),  # Chunk position info
                        "content": str(chunk["content"])
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
                            # First stage - 64D retrieval (high recall, low precision)
                            Prefetch(
                                query=dense_vector[:64],
                                using="matryoshka_64",
                                limit=100,
                            ),
                        ],
                        # Second stage - 128D refinement
                        query=dense_vector[:128],
                        using="matryoshka_128",
                        limit=50,
                    ),
                ],
                # Third stage - 256D final refinement
                query=dense_vector[:256],
                using="matryoshka_256",
                limit=25,
            )

            # **Step B: Integer-Based Quantized Search & 768D Dense Re-ranking**
            quantized_dense_prefetch = Prefetch(
                prefetch=[
                    # Integer-based search for speed
                    Prefetch(
                        query=quantized_query,
                        using="quantized",
                        limit=100,
                    )
                ],
                # Refine with full precision 768D float vectors
                query=dense_vector,
                using="dense",
                limit=25,
            )

            # **Step C: Sparse Search (BM25 for text, TBD for images)**
            sparse_search_prefetch = Prefetch(
                query=SparseVector(
                    indices=sparse_vector["indices"],
                    values=sparse_vector["values"]
                ),
                using="sparse_text",
                limit=25,
            )

            # **Step D: Fusion of Quantized Dense + Sparse using RRF**
            sparse_dense_rrf = Prefetch(
                prefetch=[quantized_dense_prefetch, sparse_search_prefetch],
                query=FusionQuery(fusion=FusionQuery.RRF),
            )

            # **Execute Hybrid Search Pipeline**
            results = await qdrant_session.client.query_points(
                collection_name=user_id,
                prefetch=[matryoshka_prefetch, sparse_dense_rrf],
                query=dense_vector,
                using="dense",
                limit=top_k,
                search_params={"hnsw_ef": 128, "exact": False},
                with_payload=True,
                query_filter=query_filter
            )

            # **Extract Documents for Reranking**
            documents = [res.payload["content"] for res in results if "content" in res.payload]

            if not documents:
                logging.warning("No documents retrieved for reranking.")
                return results  # Return original results if no content found

            # **Step E: Apply Late Interaction Reranking (ColBERT-V2)**
            reranked_results = await self.rerank_with_colbert(query_text, documents, results)

            return reranked_results

        except Exception as e:
            logging.error(f"Hybrid search failed for user {user_id}: {str(e)}")
            return []

    async def rerank_with_colbert(self, query: str, documents: List[str], results: List[Dict]) -> List[Dict]:
        """
        Uses ColBERT-V2 for late interaction reranking of retrieved documents.

        Args:
            query (str): User query.
            documents (List[str]): List of retrieved documents.
            results (List[Dict]): Initial hybrid search results.

        Returns:
            List[Dict]: Reranked search results based on ColBERT scoring.
        """
        try:
            # Initialize ColBERT Config
            colbert_config = ColBERTConfig(query_maxlen=32, doc_maxlen=512)
            ckpt = Checkpoint("jinaai/jina-colbert-v1-en", colbert_config=colbert_config)

            # Encode Query & Documents
            query_vector = ckpt.queryFromText([query])
            document_vectors = ckpt.docFromText(documents, bsize=32)

            # Compute Scores using Late Interaction
            scores = (query_vector @ document_vectors.transpose(1, 2)).max(dim=2).values.sum(dim=1).cpu().numpy()

            # Rank Documents based on ColBERT Scores
            ranked_indices = np.argsort(scores)[::-1]

            # Reorder Results Based on Ranking
            reranked_results = [results[i] for i in ranked_indices]

            return reranked_results

        except Exception as e:
            logging.error(f"ColBERT reranking failed: {str(e)}")
            return results