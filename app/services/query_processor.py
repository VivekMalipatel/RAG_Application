import asyncio
import logging
from typing import Dict, Any, List
from app.core.embedding.embedding_handler import EmbeddingHandler
from app.core.vector_store.qdrant.qdrant_handler import QdrantHandler
from app.core.models.ollama.ollama import OllamaClient
from app.config import settings

class QueryProcessor:
    """Processes user queries using hybrid search and LLM-based answering."""

    def __init__(self):
        """Initializes query processing components."""
        self.qdrant = QdrantHandler()
        self.embedding_model = EmbeddingHandler(
            model_source=settings.TEXT_EMBEDDING_SOURCE,
            model_name=settings.TEXT_EMBEDDING_MODEL,
            model_type="text"
        )
        self.ollama = OllamaClient(
            hf_repo=settings.TEXT_LLM_MODEL,
            quantization=settings.TEXT_LLM_QUANTIZATION,
            system_prompt="""
            You are an AI assistant capable of answering queries using retrieved information.
            Given relevant search results, generate a concise and factual response.
            - Use only the retrieved content.
            - Avoid making assumptions or hallucinating information.
            - If uncertain, say "Not enough information available."
            """,
            temperature=float(settings.TEXT_LLM_TEMPERATURE),
            top_p=float(settings.TEXT_LLM_TOP_P),
            max_tokens=int(settings.TEXT_LLM_MAX_TOKENS),
        )

    async def process_query(self, user_id: str, query: str, top_k: int = 10) -> Dict[str, Any]:
        """
        Processes a user query through embedding-based retrieval and LLM answering.

        Args:
            user_id (str): User's unique identifier for search.
            query (str): The query text.
            top_k (int, optional): Number of top search results. Defaults to 10.

        Returns:
            Dict[str, Any]: LLM-generated answer with supporting sources.
        """
        try:
            logging.info(f"Processing query for user {user_id}: {query}")

            # Step 1: Generate Dense & Sparse Embeddings
            dense_embedding_task = self.embedding_model.encode_dense(query)
            sparse_embedding_task = self.embedding_model.encode_sparse(query)
            dense_embedding, sparse_embedding = await asyncio.gather(dense_embedding_task, sparse_embedding_task)

            # Step 2: Perform Hybrid Search in Qdrant
            search_results = await self.qdrant.hybrid_search(
                user_id=user_id,
                query_text=query,
                dense_vector=dense_embedding,
                sparse_vector=sparse_embedding,
                top_k=top_k
            )

            if not search_results:
                return {"answer": "No relevant information found in the database.", "sources": []}

            # Step 3: Prepare Context for LLM
            retrieved_texts = [res.payload["content"] for res in search_results if hasattr(res, "payload") and res.payload and "content" in res.payload]
            context_text = "\n\n".join(retrieved_texts)

            prompt = f"""
            **User Query:** {query}

            **Retrieved Context:**
            {context_text}

            **Instructions:**
            - Answer based strictly on the retrieved content.
            - If the context is insufficient, state: "Not enough information available."
            - Ensure factual accuracy and avoid speculation.

            **Response:**
            """

            # Step 4: Generate Answer using LLM
            answer = await self.ollama.generate(prompt)

            # Step 5: Format and Return Response
            return {
                "answer": answer.strip(),
                "sources": [{"content": res.payload["content"], "file_name": res.payload["file_name"]}
                            for res in search_results if hasattr(res, "payload") and res.payload and "content" in res.payload]
            }

        except Exception as e:
            logging.error(f"Query processing failed: {str(e)}")
            return {"answer": "An error occurred while processing the query.", "sources": []}