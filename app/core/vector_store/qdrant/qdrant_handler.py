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
import asyncio

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
        sparse_enabled: bool = True,
        force_recreate: bool = False
    ):
        """Creates a user-specific collection in Qdrant with a retry mechanism."""
        max_retries = 5  # Maximum number of retry attempts
        retry_delay = 2  # Initial delay in seconds (exponential backoff)

        for attempt in range(max_retries):
            try:
                if not user_id:
                    raise ValueError("user_id cannot be empty")
                
                user_id = str(user_id)
                # Check if collection already exists
                collections = await qdrant_session.client.get_collections()
                collection_exists = user_id in [c.name for c in collections.collections]

                # If collection exists and force_recreate is False, skip creation
                if collection_exists and not force_recreate:
                    logging.info(f"Collection for user {user_id} already exists, skipping creation")
                    return

                # Delete collection if it exists and we're forcing recreation
                if collection_exists and force_recreate:
                    logging.info(f"Force recreating collection for user {user_id}")
                    await qdrant_session.client.delete_collection(collection_name=user_id)

                # Define vector configurations
                vectors_config = {
                    "dense": VectorParams(
                        size=dense_vector_size,
                        distance=models.Distance.COSINE,
                        on_disk=True
                    ),
                    "quantized": VectorParams(
                        size=quantized_size,
                        distance=models.Distance.COSINE,
                        on_disk=True
                    )
                }

                # Add Matryoshka embeddings
                for dim in matryoshka_sizes:
                    vectors_config[f"matryoshka_{dim}"] = VectorParams(
                        size=dim,
                        distance=models.Distance.COSINE,
                        on_disk=False
                    )

                # Add Sparse embeddings (BM25 for text)
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
                await qdrant_session.client.create_collection(
                    collection_name=user_id,
                    vectors_config=vectors_config,
                    sparse_vectors_config=sparse_vectors_config,
                    optimizers_config=optimizers_config,
                    on_disk_payload=True
                )

                logging.info(f"Created hybrid search collection for user {user_id}")
                return  # Success, exit the retry loop

            except ValueError as ve:
                logging.error(f"Validation error creating collection for user {user_id}: {str(ve)}")
                raise
            except Exception as e:
                logging.error(f"Failed to create collection for user {user_id} (Attempt {attempt+1}): {str(e)}")
                
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    logging.info(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    logging.critical("Max retries reached. Collection creation failed.")
                    raise  # Raise the exception after max retries


    async def store_document_vectors(self, embedded_chunks: List[Dict[str, Any]], user_id: str):
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
                        "relationships": metadata.get("relationships"),
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

    async def store_chat_vectors(self, embedded_payload: List[Dict[str, Any]], user_id: str):
        """
        Stores chat message vectors with their embeddings.

        Args:
            embedded_payload (list): List of chat messages with embeddings.
            user_id (str): User ID for the collection.
        """
        try:
            # Check if collection exists, if not create it
            collections = await qdrant_session.client.get_collections()
            if str(user_id) not in [c.name for c in collections.collections]:
                await self.create_collection(user_id=user_id)

            points = []

            for chat in embedded_payload:
                # Validate Dense Embedding Dimension
                if len(chat["dense_embedding"]) != 768:
                    raise ValueError(f"Dense vector dimension mismatch. Expected 768, got {len(chat['dense_embedding'])}")

                # Generate a unique identifier for the chat message
                chat_id = str(uuid.uuid4())

                # Create quantized and matryoshka embeddings
                dense_embedding = np.array(chat["dense_embedding"])
                quantized_dense_embedding = np.clip(
                    (dense_embedding * 127).astype(np.int8), -128, 127
                ).tolist()
                matryoshka_64 = dense_embedding[:64].tolist()
                matryoshka_128 = dense_embedding[:128].tolist()
                matryoshka_256 = dense_embedding[:256].tolist()

                # Define the point structure with all embeddings
                point = PointStruct(
                    id=chat_id,
                    vector={
                        "dense": chat["dense_embedding"],
                        "matryoshka_64": matryoshka_64,
                        "matryoshka_128": matryoshka_128,
                        "matryoshka_256": matryoshka_256,
                        "quantized": quantized_dense_embedding,
                        "sparse": chat["sparse_embedding"]
                    },
                    payload={
                        "chat_id": chat["chat_id"],
                        "user_id": user_id,
                        "message_type": chat["message_type"],
                        "timestamp": chat["timestamp"].isoformat() if hasattr(chat["timestamp"], "isoformat") else chat["timestamp"],
                        "entities": chat["entities"],
                        "relationships": chat["relationships"],
                        "chat_summary": chat["chat_summary"],
                        "content": chat["message"],  # Duplicate for consistency with document vectors
                        "is_chat": True  # Flag to distinguish from document vectors
                    }
                )
                points.append(point)

            # Upsert the processed chunks into Qdrant
            await qdrant_session.client.upsert(
                collection_name=user_id,
                points=points
            )
            logging.info(f"Stored {len(points)} chat messages with embeddings for user {user_id}")

        except Exception as e:
            logging.error(f"Failed to store chat vectors: {str(e)}")
            raise

    async def hybrid_search(
        self,
        user_id: str,
        query_text: str,
        dense_vector: List[float],
        sparse_vector: Dict[str, List[float]],
        image_embedding: Optional[List[float]] = None,
        top_k: int = 10,
        search_params: Optional[Dict[str, Any]] = None,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Performs dynamic hybrid search using Matryoshka embeddings, quantized dense search, sparse search, and late interaction reranking.

        Args:
            user_id (str): Collection name.
            query_text (str): Original text query.
            dense_vector (List[float]): Full 768D dense embedding of the query.
            sparse_vector (Dict[str, List[float]]): Sparse vector representation (BM25 for text).
            image_embedding (Optional[List[float]]): Optional image embedding for multimodal search.
            top_k (int): Number of results to return.
            search_params (Optional[Dict]): Dynamic search parameters.
            filters (Optional[Dict]): Additional search filters.

        Returns:
            List[Dict]: Final ranked results.
        """
        try:
            query_filter = Filter(**filters) if filters else None

            # Quantized vector for integer-based search
            quantized_query = np.clip(
                (np.array(dense_vector) * 127).astype(np.int8), -128, 127
            ).tolist()

            # **Matryoshka Dimensionality Reduction Search (Adjustable Limits)**
            matryoshka_prefetch = Prefetch(
                prefetch=[
                    Prefetch(
                        prefetch=[
                            Prefetch(
                                prefetch=[
                                    Prefetch(
                                        query=dense_vector[:64],
                                        using="matryoshka_64",
                                        limit=search_params["matryoshka_64_limit"],
                                    ),
                                ],
                                query=dense_vector[:128],
                                using="matryoshka_128",
                                limit=search_params["matryoshka_128_limit"],
                            ),
                        ],
                        query=dense_vector[:256],
                        using="matryoshka_256",
                        limit=search_params["matryoshka_256_limit"],
                    ),
                ],
                query=dense_vector,
                using="dense",
                limit=search_params["dense_limit"],
            )

            # **Quantized Search & 768D Dense Refinement**
            quantized_dense_prefetch = Prefetch(
                prefetch=[
                    Prefetch(
                        query=quantized_query,
                        using="quantized",
                        limit=search_params["quantized_limit"],
                    )
                ],
                query=dense_vector,
                using="dense",
                limit=search_params["dense_limit"],
            )

            # **Sparse Search (BM25)**
            sparse_search_prefetch = Prefetch(
                query=sparse_vector if isinstance(sparse_vector, SparseVector) else SparseVector(
                    indices=sparse_vector["indices"],
                    values=sparse_vector["values"]
                ),
                using="sparse",
                limit=search_params["sparse_limit"],
            )

            # **Fusion of Quantized Dense + Sparse using RRF**
            sparse_dense_rrf = Prefetch(
                prefetch=[quantized_dense_prefetch, sparse_search_prefetch],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
            )

            # **Execute Hybrid Search Pipeline**
            search_result = await qdrant_session.client.query_points(
                collection_name=str(user_id),
                prefetch=[matryoshka_prefetch, sparse_dense_rrf],
                query=dense_vector,
                using="dense",
                limit=search_params["final_limit"],
                search_params={"hnsw_ef": search_params["hnsw_ef"], "exact": True},
                with_payload=True,
                query_filter=query_filter
            )

            results = search_result.points

            documents = [res.payload["content"] for res in results if hasattr(res, "payload") and res.payload and "content" in res.payload]

            # **Apply Late Interaction Reranking (ColBERT-V2)**
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
    
    async def get_all_containers(self) -> List[str]:
        """
        Retrieves all user collections in Qdrant.
        
        Returns:
            List[str]: A list of collection names representing users.
        """
        try:
            collections = await qdrant_session.client.get_collections()
            user_collections = [c.name for c in collections.collections]
            logging.info(f"Retrieved {len(user_collections)} user collections from Qdrant.")
            return user_collections
        except Exception as e:
            logging.error(f"Failed to fetch user collections from Qdrant: {str(e)}")
            return []
    
    async def delete_collection(self, user_id: str):
        """
        Deletes a user-specific collection in Qdrant."
        """
        try:
            await qdrant_session.client.delete_collection(collection_name=user_id)
            logging.info(f"Deleted collection for user {user_id}")
        except Exception as e:
            logging.error(f"Failed to delete collection for user {user_id}: {str(e)}")
            raise
    
    async def get_collection_chunk_count(
        self, 
        user_id: str, 
        filters: Optional[Dict] = None
    ) -> int:
        """
        Retrieves the number of chunks stored in a user's collection,
        optionally filtered by specific criteria.
        
        Args:
            user_id (str): The user ID / collection name.
            filters (Optional[Dict]): Optional filters to count only specific chunks.
            
        Returns:
            int: The number of chunks/points in the collection.
        """
        try:
            # Check if collection exists
            collections = await qdrant_session.client.get_collections()
            if str(user_id) not in [c.name for c in collections.collections]:
                logging.warning(f"Collection for user {user_id} does not exist.")
                return 0
            
            if filters:
                # If filters are provided, we need to count using search with filter
                query_filter = Filter(**filters)
                count_result = await qdrant_session.client.count(
                    collection_name=str(user_id),
                    count_filter=query_filter
                )
                count = count_result.count
            else:
                # Get collection information which includes total count
                collection_info = await qdrant_session.client.get_collection(collection_name=str(user_id))
                count = collection_info.points_count
            
            logging.info(f"Retrieved chunk count for user {user_id}: {count}")
            return count
        except Exception as e:
            logging.error(f"Failed to get chunk count for user {user_id}: {str(e)}")
            return 0