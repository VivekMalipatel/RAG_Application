import logging
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from app.core.agent.base_agent import BaseAgent
from app.core.vector_store.qdrant.qdrant_handler import QdrantHandler
from app.config import settings

class SearchParams(BaseModel):
    """
    Pydantic schema for LLM-decided search parameters.
    """
    matryoshka_64_limit: int
    matryoshka_128_limit: int
    matryoshka_256_limit: int
    dense_limit: int
    quantized_limit: int
    sparse_limit: int
    final_limit: int
    hnsw_ef: int

class HybridSearchAgent(BaseAgent):
    """
    Agent that dynamically configures and executes hybrid search in Qdrant using LLM-based parameter tuning.
    """

    def __init__(self):
        system_prompt = """
                You are a search optimization expert. Given the query complexity and collection size, determine the optimal Qdrant search parameters.
                
                Here is how the search pipeline works based on your parameters:
                1. MATRYOSHKA EMBEDDING SEARCH: Uses a multi-stage approach with progressive dimensionality:
                    - First retrieves matryoshka_64_limit results using just 64-dimensions (fastest, roughest)
                    - Then filters to matryoshka_128_limit results using 128-dimensions
                    - Then filters to matryoshka_256_limit results using 256-dimensions
                    - Finally filters to dense_limit results using all 768-dimensions
                2. QUANTIZED VECTOR SEARCH: Retrieves quantized_limit results using 8-bit integers for faster computation
                3. SPARSE SEARCH: Retrieves sparse_limit results using BM25 keyword-matching algorithm
                4. FUSION & RERANKING: Combines all results using Reciprocal Rank Fusion.               
                5. HNSW PARAMETER: The hnsw_ef controls search recall vs. speed tradeoff (higher = better recall but slower)
                6. Final Limit: The final number of results to return after reranking
                7. Late Interaction Reranking: Then applies ColBERT-V2 late interaction reranking to rank all the retrieved Final results number of chunks. (Note : ColBERT-V2 only can process 8000 tokens at a time, so you need to optmise the number of token in this stage to be within this)          
                8. Then we will return the the top {top_k} results to the user.

                For complex queries about specific topics, consider higher sparse_limit.
                For conceptual or semantic queries, emphasize dense vectors with higher dense_limit.
                For large collections, larger matryoshka limits help maintain recall.

                Ground Rules:
                - Number of token in List[<Final Results>] should not exceed the context window of ColBERT-V2
                - Each paramenter should be a positive integer
                - No *_limit should exceed the Collection size

                You must return a JSON object with these keys:
                - matryoshka_64_limit (int): How many results to retrieve in the first stage.
                - matryoshka_128_limit (int): How many results to retrieve in the second stage.
                - matryoshka_256_limit (int): How many results to retrieve in the third stage.
                - dense_limit (int): The final number of dense results before reranking.
                - quantized_limit (int): Number of results to retrieve using quantized search.
                - sparse_limit (int): Number of sparse search (BM25) results.
                - hnsw_ef (int): The HNSW parameter controlling recall vs speed. (Recommended range: 128-512)
                _ final_limit (int): The final number of results to return after reranking.

                The response should be a valid JSON object only.
            """
        super().__init__(agent_name="HybridSearchAgent", system_prompt=system_prompt, temperature=0.7, top_p=0.95)
        self.qdrant_handler = QdrantHandler()

    async def determine_search_params(self, query_text: str, num_chunks: int, top_k: int) -> SearchParams:
        """
        Uses an LLM to determine the optimal search parameters based on query complexity and collection size.

        Args:
            query_text (str): The input search query.
            num_chunks (int): The total number of chunks in the collection.

        Returns:
            SearchParams: Search parameters dynamically decided by the LLM.
        """
        prompt = f"""
        Query: 
        
        <START QUERY>
        "{query_text}"
        <END QUERY>

        Collection size (number of chunks): {num_chunks}

        Approximate size of each chunk : {settings.TEXT_CHUNK_SIZE + settings.TEXT_CHUNK_OVERLAP}

        Number of results to the querier is expeting to return (top_k): {top_k}
        """

        params_dict = await self.generate_structured_response(prompt, schema=SearchParams)

        if not params_dict:
            logging.warning("[HybridSearchAgent] LLM failed to generate search parameters, falling back to default values.")
            return SearchParams(
                matryoshka_64_limit=min(500, num_chunks // 10),
                matryoshka_128_limit=min(400, num_chunks // 15),
                matryoshka_256_limit=min(300, num_chunks // 20),
                dense_limit=min(200, num_chunks // 25),
                quantized_limit=min(300, num_chunks // 30),
                sparse_limit=min(100, num_chunks // 50),
                hnsw_ef=256,
                final_limit=10
            )

        return params_dict

    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes a dynamic hybrid search.

        Args:
            inputs (Dict[str, Any]): Inputs containing query text and embeddings.

        Returns:
            Dict[str, Any]: Search results.
        """
        user_id = inputs.get("user_id")
        query_text = inputs.get("query_text")
        dense_vector = inputs.get("dense_vector")
        sparse_vector = inputs.get("sparse_vector")
        top_k = inputs.get("top_k", 10)

        # **Retrieve exact number of chunks in the collection**
        num_chunks = await self.qdrant_handler.get_collection_chunk_count(user_id)
        if num_chunks == 0:
            logging.warning(f"[HybridSearchAgent] Collection '{user_id}' does not exist or is empty, using default 1000 chunks.")
            num_chunks = 1000  # Fallback default

        # **Get search parameters from the LLM**
        search_params:SearchParams = await self.determine_search_params(query_text, num_chunks, top_k)

        # **Execute hybrid search**
        results = await self.qdrant_handler.hybrid_search(
            user_id=user_id,
            query_text=query_text,
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            top_k=top_k,
            search_params=search_params.model_dump()
        )

        return results