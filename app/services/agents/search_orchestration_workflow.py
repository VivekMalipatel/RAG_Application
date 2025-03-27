import asyncio
import logging
from typing import Dict, Any, List
from app.services.agents.graph_search_workflow import GraphSearchAgent
from app.services.agents.hybrid_search_workflow import HybridSearchAgent
from app.core.embedding.embedding_handler import EmbeddingHandler
from app.config import settings
from app.core.models.model_type import ModelType


class SearchOrchestrationWorkflow:
    def __init__(self):
        self.graph_agent = GraphSearchAgent()
        self.hybrid_agent = HybridSearchAgent()
        self.embedding_handler = EmbeddingHandler(
            provider=settings.TEXT_EMBEDDING_PROVIDER,
            model_name=settings.TEXT_EMBEDDING_MODEL_NAME,
            model_type=ModelType.TEXT_EMBEDDING
        )

    async def execute(self, user_id: str, query_text: str, top_k: int = 10) -> Dict[str, Any]:
        try:
            # Step 1: Generate embeddings
            dense_task = self.embedding_handler.encode_dense(query_text)
            sparse_task = self.embedding_handler.encode_sparse(query_text)
            dense_vector, sparse_vector = await asyncio.gather(dense_task, sparse_task)

            # Step 2: Parallel search
            hybrid_task = self.hybrid_agent.execute({
                "user_id": user_id,
                "query_text": query_text,
                "dense_vector": dense_vector[0],
                "sparse_vector": sparse_vector,
                "top_k": top_k
            })
            graph_task = self.graph_agent.execute({
                "user_id": user_id,
                "query_text": query_text
            })

            hybrid_results, graph_results = await asyncio.gather(hybrid_task, graph_task)

            # Step 3: Format context
            context_prompt = self._prepare_combined_context(
                query_text, hybrid_results, graph_results
            )

            return {
                "context_prompt": context_prompt,
                "sources": self._format_sources(hybrid_results),
                "graph_entities": graph_results.get("search_results", {}).get("entities", []),
                "graph_relationships": graph_results.get("search_results", {}).get("relationships", []),
                "graph_knowledge_paths": graph_results.get("search_results", {}).get("knowledge_paths", [])
            }

        except Exception as e:
            logging.exception(f"[SearchOrchestrationWorkflow] Failed during execution: {e}")
            return {
                "context_prompt": f"Query: {query_text}\n\nNo useful context could be retrieved.",
                "hybrid_sources": [],
                "graph_entities": [],
                "graph_relationships": []
            }

    def _prepare_combined_context(self, query, hybrid_results, graph_results):
        """
        Combine hybrid search results + graph search results into a single LLM-friendly prompt.
        """
        hybrid_chunks = []
        for res in hybrid_results:
            if not hasattr(res, "payload") or not res.payload:
                continue
            p = res.payload
            chunk_text = f"""
                [File: {p.get('file_name')} | Page: {p.get('page_number')} | Chunk: {p.get('chunk_number')}]

                Summary:
                {p.get('document_summary', 'N/A')}

                Context:
                {p.get('context', '')}

                Content:
                {p.get('content', '')}
            """
            hybrid_chunks.append(chunk_text.strip())

        combined_hybrid_text = "\n\n---\n\n".join(hybrid_chunks)

        # Format entity and relationship context
        entity_context = ""
        rel_context = ""
        if graph_results:
            entities = graph_results.get("search_results", {}).get("entities", [])
            relationships = graph_results.get("search_results", {}).get("relationships", [])
            paths = graph_results.get("search_results", {}).get("knowledge_paths", [])

            if entities:
                entity_context = "**Entities:**\n" + "\n".join([
                    f"- {e['text']} ({e['type']}): {e['profile']}"
                    for e in entities
                ])

            if relationships:
                rel_context = "**Relationships:**\n" + "\n".join([
                    f"- {r['source']} {r['relation_type']} {r['target']}: {r['relation_profile']}"
                    for r in relationships
                ])

        if paths:
            path_strings = []
            for i, path_data in enumerate(paths):  # Limit to top 3 paths
                entities = path_data.get('entities', [])
                relations = path_data.get('relations', [])
                
                # Build readable path string by alternating entities and relations
                path_elements = []
                for j, entity in enumerate(entities):
                    # Add entity text
                    entity_text = entity.get('text', 'Unknown Entity')
                    entity_type = entity.get('type', '')
                    path_elements.append(f"{entity_text} ({entity_type})")
                    
                    # Add relation if not the last entity
                    if j < len(relations):
                        relation_type = relations[j].get('relation_type', 'relates to')
                        path_elements.append(f"--[{relation_type}]-->")
                
                path_strings.append(f"- Path {i+1}: {' '.join(path_elements)}")
            
            path_context = "**Knowledge Paths:**\n" + "\n".join(path_strings)
        else:
            path_context = ""

        # Final prompt
        return f"""
        **User Query:** 
        <START QUERY>
        {query}
        <END QUERY>

        **Relevant Text Passages Retrieved from the database:**
        <START PASSAGES>
        {combined_hybrid_text}
        <END PASSAGES>

        **Entities and Relationships and Knowledge Paths Entracted from Knowledge graphs:**

        <START KNOWLEDGE GRAPH DATA>

        <START ENTITIES>
        {entity_context}
        <END ENTITIES>

        <START RELATIONSHIPS>
        {rel_context}
        <END RELATIONSHIPS>

        <START PATHS>
        {path_context}
        <END PATHS>

        <END KNOWLEDGE GRAPH DATA>

        **Response:**
        """

    def _format_sources(self, hybrid_results) -> List[Dict[str, str]]:
        return [
            {
                "file_name": r.payload.get("file_name", "")
            }
            for r in hybrid_results
            if hasattr(r, "payload") and r.payload
        ]