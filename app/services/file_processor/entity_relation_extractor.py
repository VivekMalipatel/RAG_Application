import logging
import json
from typing import List, Dict, Any
from app.core.graph_db.neo4j.neo4j_handler import Neo4jHandler
from langchain_experimental.graph_transformers import LLMGraphTransformer
from app.core.cache.redis_cache import RedisCache
from app.core.models.model_handler import ModelRouter
from app.core.models.model_provider import Provider
from app.core.models.model_type import ModelType
from app.config import settings
from pydantic import BaseModel


class EntitySchema(BaseModel):
    id: str
    text: str
    label: str

class RelationshipSchema(BaseModel):
    source: str
    target: str
    type: str
    confidence: float

class EntityRelationSchema(BaseModel):
    entities: List[EntitySchema]
    relationships: List[RelationshipSchema]


class EntityRelationExtractor:
    """
    Extracts structured knowledge graph entities & relationships from text chunks
    and syncs them with Neo4j, ensuring consistency and proper entity linking.
    """

    def __init__(self):
        logging.info("Initializing EntityRelationExtractor...")

        self.graph = Neo4jHandler()

        self.system_prompt = self._build_system_prompt()

        self.llm = ModelRouter(
            provider=Provider(settings.TEXT_CONTEXT_LLM_PROVIDER),
            model_name=settings.TEXT_CONTEXT_LLM_MODEL_NAME,
            model_quantization=settings.TEXT_CONTEXT_LLM_QUANTIZATION,
            model_type=ModelType.TEXT_GENERATION,
            system_prompt=self.system_prompt,
        )
    
    def _build_system_prompt(self):
        """Constructs the system prompt with examples."""
        examples = [
            {
                "text": "Adam is a software engineer at Microsoft since 2009.",
                "head": "Adam",
                "head_type": "Person",
                "relation": "WORKS_FOR",
                "tail": "Microsoft",
                "tail_type": "Company",
            },
            {
                "text": "Adam won the Best Talent award last year.",
                "head": "Adam",
                "head_type": "Person",
                "relation": "HAS_AWARD",
                "tail": "Best Talent",
                "tail_type": "Award",
            },
            {
                "text": "Microsoft produces Microsoft Word.",
                "head": "Microsoft Word",
                "head_type": "Product",
                "relation": "PRODUCED_BY",
                "tail": "Microsoft",
                "tail_type": "Company",
            },
            {
                "text": "Microsoft Word is a lightweight app that is accessible offline.",
                "head": "Microsoft Word",
                "head_type": "Product",
                "relation": "HAS_CHARACTERISTIC",
                "tail": "lightweight app",
                "tail_type": "Characteristic",
            }
        ]

        examples_str = "\n".join(
            [
                f'Text: "{ex["text"]}"\n'
                f'Entities: [{ex["head"]} ({ex["head_type"]}), {ex["tail"]} ({ex["tail_type"]})]\n'
                f'Relationship: {ex["head"]} -[{ex["relation"]}]-> {ex["tail"]}\n'
                for ex in examples
            ]
        )

        return f"""
        # Knowledge Graph Extraction Guide

        ## 1. Overview
        You are an AI specializing in **knowledge graph extraction**. Your goal
        is to **extract structured entities and their relationships** from text
        while ensuring consistency and correctness.

        ## 2. Entity Guidelines
        - Extract named entities such as **Persons, Organizations, Locations, Products, etc.**.
        - Ensure **consistent labeling** (e.g., "Person", "Company", "Product").
        - Node IDs should be **human-readable** and not integers.

        ## 3. Relationship Guidelines
        - Use **generalized relationship types** instead of overly specific ones:
          - `WORKS_FOR` instead of `EMPLOYED_AT`
          - `LOCATED_IN` instead of `BASED_IN`
          - `PRODUCED_BY` instead of `CREATED_BY`
        - Relationships must be **consistent & verifiable from the text**.
        - Assign a **confidence score (0 to 1)** to each relationship based on how certain it is.
        - **High confidence (0.9 - 1.0)**: Direct statements like “X works at Y”
        - **Medium confidence (0.5 - 0.8)**: Inferred relationships with indirect wording
        - **Low confidence (0.0 - 0.4)**: Highly uncertain relationships

        ## 4. Coreference Resolution
        - If an entity has multiple references (e.g., "Elon Musk", "Musk", "he"),
          always **use the most complete identifier** found in the text.

        ## 5. Examples
        {examples_str}

        ## 6. JSON Output Format
        ```
        {{
            "entities": [
                {{"id": "unique_id", "text": "entity text", "label": "PERSON/ORG/LOCATION/etc."}}
            ],
            "relationships": [
                {{"source": "entity1_id", "target": "entity2_id", "type": "RELATIONSHIP_TYPE"}}
            ]
        }}
        ```
        """

    async def extract_entities_and_relationships(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extracts structured knowledge graph entities & relationships from document chunks,
        ensures entity resolution, and stores them in Neo4j.
        """

        merged_chunks = self._merge_chunks(chunks)
        processed_chunks = []

        for merged_text, chunk_ids in merged_chunks:
            chunk_metadata = chunks[chunk_ids[0]]["chunk_metadata"]

            # Extract user & document metadata
            user_id = chunk_metadata["user_id"]
            document_id = chunk_metadata["document_id"]

            # Generate structured entity-relationship extraction prompt
            extraction_prompt = f"""
            Ensure that:
            - Entities are **correctly labeled** (PERSON, ORG, LOCATION, PRODUCT, etc.).
            - Relationships follow the **standardized types** as outlined in the guidelines.

            Extract structured knowledge graph entities & relationships from the following text:

            {merged_text}
            """

            # Run structured extraction through LLM
            extracted_data = await self.llm.client.generate_structured_output(extraction_prompt, schema=EntityRelationSchema)
            if "error" in extracted_data:
                logging.error(f"LLM Extraction Error: {extracted_data['error']}")
                continue

            # Extract entities & relationships
            entities = extracted_data.entities
            relationships = extracted_data.relationships

            if len(entities) > 0:

                # Convert Entity Objects to List of Dictionaries
                entities_data: List[Dict[str, Any]] = [
                    {"id": entity.id, "text": entity.text, "label": entity.label} for entity in entities
                ]

                # Convert Relationship Objects to List of Dictionaries
                relationships_data: List[Dict[str, Any]] = [
                    {
                        "source": relationship.source,
                        "target": relationship.target,
                        "type": relationship.type,
                        "confidence": relationship.confidence,
                    }
                    for relationship in relationships
                ]

                await self.graph.store_entities(entities_data, user_id, document_id)
                await self.graph.store_relationships(relationships_data, user_id)

            # Assign extracted data to all related chunks
            for chunk_id in chunk_ids:
                chunks[chunk_id]["chunk_metadata"]["entities"] = entities
                chunks[chunk_id]["chunk_metadata"]["relationships"] = relationships
                processed_chunks.append(chunks[chunk_id])

        return processed_chunks

    def _merge_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merges continuation chunks for improved entity extraction.
        """
        merged_chunks = []
        buffer_text = ""
        buffer_chunk_ids = []

        for chunk in chunks:
            chunk_text = chunk["content"]
            chunk_id = chunk["chunk_metadata"]["chunk_number"]

            if chunk["chunk_metadata"].get("is_continuation") and buffer_text:
                buffer_text += " " + chunk_text
                buffer_chunk_ids.append(chunk_id)
            else:
                if buffer_text:
                    merged_chunks.append((buffer_text, buffer_chunk_ids))
                buffer_text = chunk_text
                buffer_chunk_ids = [chunk_id]

        if buffer_text:
            merged_chunks.append((buffer_text, buffer_chunk_ids))

        return merged_chunks
