import asyncio
import pytest
from app.services.agents.hybrid_search_agent import HybridSearchAgent
from app.core.embedding.embedding_handler import EmbeddingHandler
from app.config import settings
from app.core.models.model_type import ModelType
from app.core.models.model_handler import ModelRouter
from typing import Dict, Any


class QueryProcessor:
    """Processes user queries using hybrid search and LLM-based answering."""

    def __init__(self):
        """Initializes query processing components."""

        # Initialize Embedding Model using ModelRouter
        self.embedding_model = EmbeddingHandler(
            provider=settings.TEXT_EMBEDDING_PROVIDER,
            model_name=settings.TEXT_EMBEDDING_MODEL_NAME,
            model_type=ModelType.TEXT_EMBEDDING
        )

        self.agent = HybridSearchAgent()

        # Initialize LLM for Query Answering
        self.llm = ModelRouter(
            provider=settings.TEXT_LLM_PROVIDER,
            model_name=settings.TEXT_LLM_MODEL_NAME,
            model_type=ModelType.TEXT_GENERATION,
            model_quantization=settings.TEXT_LLM_QUANTIZATION,
            system_prompt="""
            You are an AI assistant capable of answering queries using retrieved information.
            Given relevant search results, generate a concise and factual response.
            - Use only the retrieved content.
            - Avoid making assumptions or hallucinating information.
            - If uncertain, say "Not enough information available."
            """,
            temperature=float(settings.TEXT_LLM_TEMPERATURE),
            top_p=float(settings.TEXT_LLM_TOP_P)
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
            # Step 1: Generate Dense & Sparse Embeddings
            dense_embedding_task = self.embedding_model.encode_dense(query)
            sparse_embedding_task = self.embedding_model.encode_sparse(query)
            dense_embedding, sparse_embedding = await asyncio.gather(dense_embedding_task, sparse_embedding_task)

            # Step 2: Run Hybrid Search Agent
            search_results = await self.agent.execute({
                "user_id": user_id,
                "query_text": query,
                "dense_vector": dense_embedding,
                "sparse_vector": sparse_embedding,
                "top_k": top_k
            })

            if not search_results or "search_results" not in search_results or not search_results["search_results"]:
                return {"answer": "No relevant information found in the database.", "sources": []}

            # Step 3: Prepare Context for LLM
            retrieved_texts = [res.payload["content"] for res in search_results["search_results"]
                               if hasattr(res, "payload") and res.payload and "content" in res.payload]
            context_text = "\n\n".join(retrieved_texts)

            prompt = f"""
            **User Query:** {query}

            **Relevant Information Retrieved:**
            {context_text}

            **Response:**
            """

            # Step 4: Generate Answer using LLM
            answer = await self.llm.generate_text(prompt)

            # Step 5: Format and Return Response
            return {
                "answer": answer.strip(),
                "sources": [{"content": res.payload["content"], "file_name": res.payload["file_name"]}
                            for res in search_results["search_results"]
                            if hasattr(res, "payload") and res.payload and "content" in res.payload]
            }

        except Exception as e:
            return {"answer": "An error occurred while processing the query.", "sources": []}


# **🧪 Pytest Unit Test for HybridSearchAgent**
@pytest.mark.asyncio
async def test_hybrid_search_agent():
    """Tests HybridSearchAgent with a sample query."""
    
    query_processor = QueryProcessor()
    user_id = "test_user"
    query = "What are the latest AI advancements?"
    top_k = 5

    # Run the search and answer process
    response = await query_processor.process_query(user_id, query, top_k)

    # **Assertions**
    assert isinstance(response, dict), "Response should be a dictionary"
    assert "answer" in response, "Response should contain an answer field"
    assert "sources" in response, "Response should contain a sources field"
    assert isinstance(response["sources"], list), "Sources should be a list"
    
    # Ensure LLM response is not empty
    assert response["answer"], "Answer should not be empty"

    # Print the result for debugging
    print("\nTest Output:\n", response)


# **🧪 Manual Test Runner**
if __name__ == "__main__":
    query_processor = QueryProcessor()
    query = input("Enter your query: ")
    top_k = int(input("Enter the number of results you want: "))
    user_id = 'test_user'

    response = asyncio.run(query_processor.process_query(user_id, query, top_k))
    print("\nFinal Output:\n", response)