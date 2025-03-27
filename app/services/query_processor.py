import asyncio
import logging
from typing import Dict, Any, List
from app.core.embedding.embedding_handler import EmbeddingHandler
from app.core.vector_store.qdrant.qdrant_handler import QdrantHandler
from app.core.models.model_handler import ModelRouter
from app.core.models.model_provider import Provider
from app.services.agents.hybrid_search_workflow import HybridSearchAgent
from app.core.models.model_type import ModelType
from app.config import settings

# class QueryProcessor:
#     """Processes user queries using hybrid search and LLM-based answering."""

#     def __init__(self):
#         """Initializes query processing components."""
#         self.qdrant = QdrantHandler()

#         # Initialize Embedding Model using ModelRouter
#         self.embedding_model = EmbeddingHandler(
#             provider=settings.TEXT_EMBEDDING_PROVIDER,
#             model_name=settings.TEXT_EMBEDDING_MODEL_NAME,
#             model_type=ModelType.TEXT_EMBEDDING
#         )

#         self.agent = HybridSearchAgent()

#         # Initialize LLM for Query Answering
#         self.llm = ModelRouter(
#             provider=settings.TEXT_LLM_PROVIDER,
#             model_name=settings.TEXT_LLM_MODEL_NAME,
#             model_type=ModelType.TEXT_GENERATION,
#             model_quantization=settings.TEXT_LLM_QUANTIZATION,
#             system_prompt="""
#             You are an AI assistant capable of answering queries using retrieved information.
#             Given relevant search results, generate a concise and factual response.
#             - Provide a clear and coherent answer.
#             - Answer from whatever information is available, even if it is limited.
#             - If the Information is not sufficient, just answer based on the available information.
#             """,
#             temperature=float(settings.TEXT_LLM_TEMPERATURE),
#             top_p=float(settings.TEXT_LLM_TOP_P)
#         )

#     async def process_query(self, user_id: str, query: str, top_k: int = 10) -> Dict[str, Any]:
#         """
#         Processes a user query through embedding-based retrieval and LLM answering.

#         Args:
#             user_id (str): User's unique identifier for search.
#             query (str): The query text.
#             top_k (int, optional): Number of top search results. Defaults to 10.

#         Returns:
#             Dict[str, Any]: LLM-generated answer with supporting sources.
#         """
#         try:
#             logging.info(f"Processing query for user {user_id}: {query}")

#             # Step 1: Generate Dense & Sparse Embeddings
#             dense_embedding_task = self.embedding_model.encode_dense(query)
#             sparse_embedding_task = self.embedding_model.encode_sparse(query)
#             dense_embedding, sparse_embedding = await asyncio.gather(dense_embedding_task, sparse_embedding_task)

#             # Step 2: Perform Hybrid Search in Qdrant
#             """
#             search_results = await self.qdrant.hybrid_search(
#                 user_id=user_id,
#                 query_text=query,
#                 dense_vector=dense_embedding,
#                 sparse_vector=sparse_embedding,
#                 top_k=top_k
#             )
#             """
#             search_results = await self.agent.execute({
#                 "user_id": user_id,
#                 "query_text": query,
#                 "dense_vector": dense_embedding[0],
#                 "sparse_vector": sparse_embedding,
#                 "top_k": top_k
#             })

#             if not search_results:
#                 return {"answer": "No relevant information found in the database.", "sources": []}

#             # Step 3: Prepare Context for LLM
#             retrieved_texts = [res.payload["content"] for res in search_results if hasattr(res, "payload") and res.payload and "content" in res.payload]
#             context_text = "\n\n".join(retrieved_texts)

#             prompt = f"""
#             **User Query:** {query}

#             **Relevant information::**
#             {context_text}

#             **Response:**
#             """

#             # Step 4: Generate Answer using LLM
#             answer = await self.llm.generate_text(prompt)

#             # Step 5: Format and Return Response
#             return {
#                 "answer": answer.strip(),
#                 "sources": [{"content": res.payload["content"], "file_name": res.payload["file_name"]}
#                             for res in search_results if hasattr(res, "payload") and res.payload and "content" in res.payload]
#             }

#         except Exception as e:
#             logging.error(f"Query processing failed: {str(e)}")
#             return {"answer": "An error occurred while processing the query.", "sources": []}

from app.services.agents.search_orchestration_workflow import SearchOrchestrationWorkflow

class QueryProcessor:
    def __init__(self):
        self.search_orchestrator = SearchOrchestrationWorkflow()
        self.llm = ModelRouter(
            provider=Provider.OPENAI,
            model_name="gpt-4o-mini",
            model_type=ModelType.TEXT_GENERATION,
            #model_quantization=settings.TEXT_LLM_QUANTIZATION,
            system_prompt="""
            You are an AI assistant powered by a RAG (Retrieval-Augmented Generation) system. You will be ansnwering to user queries based on the Your answers must be based on the information provided to you in each query context.
            
            When answering:
            1. Prioritize information from retrieved documents and structured knowledge.
            2. Use entities and relationships from the knowledge graph to provide conceptual understanding.
            3. Cite specific information from retrieved documents to support your answers.
            4. Synthesize information across multiple sources when relevant.``
            5. Maintain a clear distinction between facts from retrieved content and any necessary contextual explanations.
            6. When information is incomplete, clearly acknowledge limitations rather than inventing details.
            7. Present answers in a concise, well-structured format.
            8. For complex topics, organize information logically with appropriate headings or bullet points.
            9. Ignore any unrelated information from the retrieved documents, but you may use it for general knowledge or context if applicable.
            
            Your goal is to provide accurate, helpful responses grounded entirely in the retrieved information. If no information is available, you may respond "No information found on this topic.", however, do not respond with your own knowledge or opinions.
            """,
            temperature=float(settings.TEXT_LLM_TEMPERATURE),
            top_p=float(settings.TEXT_LLM_TOP_P)
        )

    async def process_query(self, user_id: str, query: str, top_k: int = 10) -> Dict[str, Any]:
        try:
            logging.info(f"[QueryProcessor] Processing query: {query}")

            # Step 1: Unified search
            search_results = await self.search_orchestrator.execute(user_id, query, top_k)

            # Step 2: LLM Response
            answer = await self.llm.generate_text(search_results["context_prompt"])

            return {
                "answer": answer.strip(),
                "sources": search_results["sources"],
                "entities": search_results["graph_entities"],
                "relationships": search_results["graph_relationships"],
                "knowledge_paths": search_results["graph_knowledge_paths"]
            }

        except Exception as e:
            logging.error(f"[QueryProcessor] Failed to process query: {e}")
            return {
                "answer": "An error occurred while processing the query.",
                "sources": [],
                "entities": [],
                "relationships": []
            }