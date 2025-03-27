import logging
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from app.core.agent.base_agent import BaseAgent
from app.core.graph_db.neo4j.neo4j_handler import Neo4jHandler
from app.core.graph_db.neo4j.neo4j_search import Neo4jSearchHandler
from app.services.file_processor.entity_relation_extractor import EntityRelationExtractor, EntityRelationSchema


class GraphSearchParams(BaseModel):
    """
    Pydantic schema for LLM-decided graph search parameters.
    """
    search_depth: int
    expansion_factor: int
    relevance_threshold: float
    max_paths: int
    include_metadata: bool


class GraphSearchAgent(BaseAgent):
    """
    Agent that dynamically extracts entities & relationships, converts them into dense embeddings, 
    and executes knowledge graph search in Neo4j.
    """

    def __init__(self):
        system_prompt = """
            You are an expert in graph-based information retrieval. Given a user's query and extracted entities & relationships, 
            determine the optimal Neo4j search parameters.

            The knowledge graph search involves:
            1. **Entity Similarity Search**: Find the most relevant entities based on embeddings.
            2. **Relationship Expansion**: Retrieve relationships linked to the query entities.
            3. **Path Traversal**: Explore paths between entities up to a specified depth.

            **Dynamic Optimization:**
            - Increase `search_depth` for broad conceptual queries.
            - Increase `relevance_threshold` for precise results.
            - If the graph has **high entity density**, lower `expansion_factor` to avoid overload.

            **Constraints:**
            - search_depth must be between 1 and 5.
            - expansion_factor must be between 2 and 20.
            - relevance_threshold should be between 0.2 and 1.0 (higher = more precise matches).
            - max_paths should not exceed 50.

            Return a **valid JSON object** only.
        """
        super().__init__(agent_name="GraphSearchAgent", system_prompt=system_prompt, temperature=0.7, top_p=0.95)
        self.neo4j_handler = Neo4jHandler()
        self.neo4j_search = Neo4jSearchHandler()
        self.er = EntityRelationExtractor()
        self.embedding_handler = self.er.embedding_handler
        self.er_model = self.er.llm

    async def extract_entities_and_relationships(self, query_text: str) -> EntityRelationSchema:
        """
        Uses an LLM to extract structured entities & relationships from the query text.

        Args:
            query_text (str): User's query text.

        Returns:
            EntityRelationSchema: Extracted entities and relationships.
        """
        prompt = f"""
            Extract structured knowledge graph entities & relationships from the following query:
            
            <START TEXT>
            {query_text}
            <END TEXT>

            Ensure:
            - Entities are labeled correctly (Person, Organization, Location, Product, etc.).
            - Relationships describe the connection between entities.
            - Each entity & relationship has a profile describing what might the eventity. As you donot have any context to generate a profile for this entity just generate a ground truth pofile like a PERSON, NAME, PLACE, ANIMAL, THING, etc. Donot go for factual search while doing this, as we at this point doesn't have any context to generate a profile.
        """
        extracted_data = await self.er_model.client.generate_structured_output(prompt, schema=EntityRelationSchema)

        if not extracted_data:
            logging.warning("[GraphSearchAgent] LLM failed to extract entities & relationships.")
            return EntityRelationSchema(entities=[], relationships=[])

        return extracted_data

    async def determine_graph_search_params(self, query_text: str, entity_count: int) -> GraphSearchParams:
        """
        Uses an LLM to determine the optimal graph search parameters based on query complexity and entity occurrence.

        Args:
            query_text (str): The input search query.
            entity_count (int): The number of extracted entities.

        Returns:
            GraphSearchParams: Optimized search parameters decided by the LLM.
        """
        prompt = f"""
        Query: 
        
        <START QUERY>
        "{query_text}"
        <END QUERY>

        Number of extracted entities: {entity_count}
        """
        params_dict = await self.generate_structured_response(prompt, schema=GraphSearchParams)

        if not params_dict:
            logging.warning("[GraphSearchAgent] LLM failed to generate search parameters, using defaults.")
            return GraphSearchParams(
                search_depth=min(3, max(1, entity_count // 10)),
                expansion_factor=min(10, max(2, entity_count // 5)),
                relevance_threshold=0.5,
                max_paths=20,
                include_metadata=True
            )

        return params_dict

    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extracts entities & relationships, converts them into dense embeddings, and executes knowledge graph search.

        Args:
            inputs (Dict[str, Any]): Inputs containing query text.

        Returns:
            Dict[str, Any]: Search results from Neo4j.
        """
        user_id = int(inputs.get("user_id"))
        query_text = inputs.get("query_text")

        # **Step 1: Extract Entities & Relationships**
        extracted_data: EntityRelationSchema = await self.extract_entities_and_relationships(query_text)
        entities = extracted_data.entities
        relationships = extracted_data.relationships

        if not entities and not relationships:
            logging.warning("[GraphSearchAgent] No entities or relationships extracted from query.")
            return {"error": "No relevant entities or relationships found in query."}

        entity_results, relation_results, knowledge_paths = [], [], []

        # **Step 2: Convert Entities & Relationships to Dense Embeddings**
        if entities:
            entities_data, entity_texts = [], []
            for e in entities:
                entity_text = f"{e.text} {e.entity_type} {e.entity_profile}"
                entities_data.append({
                    "id": e.id,
                    "text": e.text,
                    "entity_type": e.entity_type,
                    "entity_profile": e.entity_profile
                })
                entity_texts.append(entity_text)
            entity_embeddings = await self.embedding_handler.encode_dense(entity_texts)
            entity_embeddings = [emb[:256] if len(emb) >= 256 else emb for emb in entity_embeddings]

        if relationships:
            relationships_data, relation_texts = [], []
            for r in relationships:
                relation_text = f"{r.source} {r.target} {r.relation_type} {r.relation_profile}"
                relationships_data.append({
                    "source": r.source,
                    "target": r.target,
                    "relation_type": r.relation_type,
                    "confidence": r.confidence,
                    "relation_profile": r.relation_profile
                })
                relation_texts.append(relation_text)
            
            relation_embeddings = await self.embedding_handler.encode_dense(relation_texts)
            relation_embeddings = [emb[:256] if len(emb) >= 256 else emb for emb in relation_embeddings]

        # **Step 3: Determine Graph Search Parameters**
        search_params = await self.determine_graph_search_params(query_text, len(entities))

        # **Step 4: Perform Graph Search**
        if entities:
            for entity, embedding in zip(entities_data, entity_embeddings):
                entity_result = await self.neo4j_search.search_entities(
                    user_id=user_id,
                    query_embedding=embedding,
                    limit=search_params.expansion_factor * 2
                )          

        if relationships:
            for relation, embedding in zip(relationships_data, relation_embeddings):
                relation_result = await self.neo4j_search.search_relationships(
                    user_id=user_id,
                    query_embedding=embedding,
                    limit=search_params.max_paths
                )
            
        if entities:
            for entity in entities:
                paths = await self.neo4j_search.retrieve_knowledge_paths(
                    user_id=user_id,
                    entity_id=entity.id,
                    max_depth=search_params.search_depth,
                    limit=search_params.max_paths
                )

        # **Step 5: Return structured results**
        return {
            "extracted_data": {
            "entities": [entity.model_dump() for entity in entities] if entities else [],
            "relationships": [rel.model_dump() for rel in relationships] if relationships else []
            },
            "search_results": {
            "entities": entity_result if entities else [],
            "relationships": relation_result if relationships else [],
            "knowledge_paths": paths if entities else [],
            },
            "search_parameters": search_params.model_dump()
        }