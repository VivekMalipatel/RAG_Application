import logging
from typing import List, Dict, Any, Tuple
from app.core.graph_db.neo4j.neo4j_handler import Neo4jHandler
from app.core.embedding.embedding_handler import EmbeddingHandler
from app.core.models.model_handler import ModelRouter
from app.core.models.model_provider import Provider
from app.core.models.model_type import ModelType
from app.config import settings
from pydantic import BaseModel
import asyncio


class EntitySchema(BaseModel):
    id: str
    text: str 
    entity_type: str
    entity_profile: str

class RelationSchema(BaseModel):
    source: str 
    target: str 
    relation_type: str
    confidence: float
    relation_profile: str 

class EntityRelationSchema(BaseModel):
    entities: List[EntitySchema]
    relationships: List[RelationSchema]


class EntityRelationExtractor:
    """
    Extracts structured knowledge graph entities & relationships from text chunks
    and syncs them with Neo4j, ensuring consistency and proper entity linking.
    """

    def __init__(self):
        logging.info("Initializing EntityRelationExtractor...")

        self.graph = Neo4jHandler()

        self.embedding_handler = EmbeddingHandler(
            provider=Provider.HUGGINGFACE,
            model_name=settings.TEXT_EMBEDDING_MODEL_NAME,
            model_type=ModelType.TEXT_EMBEDDING
        )

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
                "entity_1": "Adam",
                "entity_1_type": "Person",
                "entity_1_profile": "Adam is a software engineer specializing in cloud computing and AI.",
                "relation": "WORKS_FOR",
                "relation_profile": "The WORKS_FOR relationship indicates employment at a company.",
                "entity_2": "Microsoft",
                "entity_2_type": "Company",
                "entity_2_profile": "Microsoft is a technology company known for software and cloud services."
            },
            {
                "text": "Adam won the Best Talent award last year.",
                "entity_1": "Adam",
                "entity_1_type": "Person",
                "entity_1_profile": "Adam is a recognized individual in his field, awarded for excellence in his work.",
                "relation": "HAS_AWARD",
                "relation_profile": "HAS_AWARD signifies a person receiving recognition for achievements.",
                "entity_2": "Best Talent",
                "entity_2_type": "Award",
                "entity_2_profile": "Best Talent is an annual award given to outstanding professionals."
            },
            {
                "text": "Microsoft produces Microsoft Word.",
                "entity_1": "Microsoft Word",
                "entity_1_type": "Product",
                "entity_1_profile": "Microsoft Word is a widely used document editing software.",
                "relation": "PRODUCED_BY",
                "relation_profile": "PRODUCED_BY signifies the creation or ownership of a product by a company.",
                "entity_2": "Microsoft",
                "entity_2_type": "Company",
                "entity_2_profile": "Microsoft is a multinational technology company producing software and hardware."
            },
            {
                "text": "Microsoft Word is a lightweight app that is accessible offline.",
                "entity_1": "Microsoft Word",
                "entity_1_type": "Product",
                "entity_1_profile": "Microsoft Word is a widely used document editing software with offline capabilities.",
                "relation": "HAS_CHARACTERISTIC",
                "relation_profile": "HAS_CHARACTERISTIC represents an attribute or feature associated with an entity.",
                "entity_2": "lightweight app",
                "entity_2_type": "Characteristic",
                "entity_2_profile": "A lightweight app is a software application optimized for performance and low resource usage."
            }
        ]
        examples_str = "\n".join(
            [
                f'Text: "{ex["text"]}"\n'
                f'Entities: [{ex["entity_1"]} ({ex["entity_1_type"]}) - Profile: "{ex["entity_1_profile"]}", '
                f'{ex["entity_2"]} ({ex["entity_2_type"]}) - Profile: "{ex["entity_2_profile"]}"]\n'
                f'Relationship: {ex["entity_1"]} -[{ex["relation"]}]-> {ex["entity_2"]} '
                f'(Profile: "{ex["relation_profile"]}")\n'
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
        - **Entity IDs must be unique and human-readable** to prevent duplication.
        - Generate entity IDs using the following format:
          - convert the entity text to lowercase and add and underscore if there are spaces.
          - **Format:** `"<entity_text_lowercase>"`
          - **Example:** `"adam"`, `"microsoft_corporation"`, `"best_buy"`
          - **If the same entity appears multiple times, use the first mention as the canonical ID.**
        - Avoid using **numeric or system-generated UUIDs** unless necessary.

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

        ## 4. Profiling Guidelines
        - Each entity-relation pair must include a **profile description** that provides:
        - **Entity Profile**: Describes the entity’s role and importance.
        - **Relation Profile**: Explains the meaning and relevance of the relationship.

        ## 5. Coreference Resolution
        - If an entity has multiple references (e.g., "Elon Musk", "Musk", "he"),
          always **use the most complete identifier** found in the text.

        ## 5. Examples
        {examples_str}

        ## 6. JSON Output Format
        ```
        {{
            "entities": [
                {{
                    "id": "unique_id",
                    "text": "entity text",
                    "entity_type": "PERSON/ORG/LOCATION/etc.",
                    "entity_profile": "Entity profile describing its role and importance."
                }}
            ],
            "relationships": [
                {{
                    "source": "entity1_id",
                    "target": "entity2_id",
                    "relation_type": "RELATIONSHIP_TYPE",
                    "confidence": 0.9,
                    "relation_profile": "Relationship profile explaining its significance."
                }}
            ]
        }}
        ```
        """
    
    async def extract_entities_and_relationships(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extracts structured knowledge graph entities & relationships from document chunks asynchronously,
        ensuring entity resolution and storing them in Neo4j.
        """
        merged_chunks = self._merge_chunks(chunks)
        processed_chunks = []

        async def process_chunk(merged_text, chunk_ids, merged_context):
            chunk_metadata = chunks[chunk_ids[0]]["chunk_metadata"]
            user_id = chunk_metadata["user_id"]
            document_id = chunk_metadata["document_id"]

            # Generate structured entity-relationship extraction prompt
            extraction_prompt = f"""
            Ensure that:
            - Entities are **correctly labeled** (PERSON, ORG, LOCATION, PRODUCT, etc.).
            - Each entity and relationship has a **profile description** explaining its significance.

            Extract structured knowledge graph entities & relationships from the following text:
            
            <START TEXT>
            {merged_text}
            <END TEXT>

            Additional text context for referenc (You may use the entities in this context for extraction, if required):
            <START CONTEXT>
            {merged_context}
            <END CONTEXT>

            For your context regarding the document this text is extracted from, please refer to the below document summary (Just for your context, donot extract entities from this summary until and unless you are not able to get a meaningful entities and realtions from the text and its context):
            {chunk_metadata["doc_summary"]}

            """

            # Run structured extraction through LLM
            extracted_data: EntityRelationSchema = await self.llm.client.generate_structured_output(
                extraction_prompt, schema=EntityRelationSchema
            )
            if "error" in extracted_data:
                logging.error(f"LLM Extraction Error: {extracted_data['error']}")
                return None

            # Extract entities & relationships
            entities = extracted_data.entities
            relationships = extracted_data.relationships

            # Process entities first
            if entities:
                chunk_entities_data, entities_data, entity_texts = [], [], []
                for e in entities:
                    entity_text = f"{e.text} {e.entity_type} {e.entity_profile}"
                    entities_data.append({
                        "id": e.id,
                        "text": e.text,
                        "entity_type": e.entity_type,
                        "entity_profile": e.entity_profile
                    })
                    chunk_entities_data.append({
                        "id": e.id,
                        "text": e.text,
                        "entity_type": e.entity_type,
                        "entity_profile": e.entity_profile
                    })
                    entity_texts.append(entity_text)

                # Generate embeddings for entities
                entity_embeddings = await self.embedding_handler.encode_dense(entity_texts)
                entity_embeddings = [emb[:256] if len(emb) >= 256 else emb for emb in entity_embeddings]

                # Attach embeddings to entities
                for i, entity in enumerate(entities_data):
                    entity["embedding"] = entity_embeddings[i]

                # Store entities in Neo4j
                neo4j_entity_ids = await self.graph.store_entities(entities_data, user_id, document_id)

            else:
                chunk_entities_data = []

            # Process relationships separately
            if relationships:
                chunk_relationships_data, relationships_data, relation_texts = [], [], []
                for r in relationships:
                    relation_text = f"{r.source} {r.target} {r.relation_type} {r.relation_profile}"
                    relationships_data.append({
                        "source": r.source,
                        "target": r.target,
                        "relation_type": r.relation_type,
                        "confidence": r.confidence,
                        "relation_profile": r.relation_profile
                    })
                    chunk_relationships_data.append({
                        "source": r.source,
                        "target": r.target,
                        "relation_type": r.relation_type,
                        "confidence": r.confidence,
                        "relation_profile": r.relation_profile
                    })
                    relation_texts.append(relation_text)

                # Generate embeddings for relationships
                relation_embeddings = await self.embedding_handler.encode_dense(relation_texts)
                relation_embeddings = [emb[:256] if len(emb) >= 256 else emb for emb in relation_embeddings]

                # Attach embeddings to relationships
                for i, relation in enumerate(relationships_data):
                    relation["embedding"] = relation_embeddings[i]

                # Store relationships in Neo4j
                neo4j_relationship_ids = await self.graph.store_relationships(relationships_data, user_id)

            else:
                chunk_relationships_data = []

            # Assign extracted data to all related chunks
            for chunk_id in chunk_ids:
                chunks[chunk_id]["chunk_metadata"]["entities"] = chunk_entities_data
                chunks[chunk_id]["chunk_metadata"]["relationships"] = chunk_relationships_data
                processed_chunks.append(chunks[chunk_id])

        # Run all chunk processing in parallel
        await asyncio.gather(*[process_chunk(merged_text, chunk_ids, merged_context) for merged_text, chunk_ids, merged_context in merged_chunks])

        return processed_chunks

    def _merge_chunks(self, chunks: List[Dict[str, Any]]) -> List[Tuple[str, List[int], str]]:
        """
        Merges continuation chunks while preserving their context.
        
        Returns:
            List of tuples -> (merged_text, chunk_ids, merged_context)
        """
        merged_chunks = []
        buffer_text = ""
        buffer_context = ""
        buffer_chunk_ids = []

        for chunk in chunks:
            chunk_text = chunk["content"]
            chunk_context = chunk["chunk_metadata"].get("context", "")  # Extract chunk context
            chunk_id = chunk["chunk_metadata"]["chunk_number"]

            if chunk["chunk_metadata"].get("is_continuation") and buffer_text:
                buffer_text += " " + chunk_text
                buffer_context += " " + chunk_context if chunk_context else ""
                buffer_chunk_ids.append(chunk_id)
            else:
                if buffer_text:
                    merged_chunks.append((buffer_text, buffer_chunk_ids, buffer_context))
                buffer_text = chunk_text
                buffer_context = chunk_context if chunk_context else ""
                buffer_chunk_ids = [chunk_id]

        if buffer_text:
            merged_chunks.append((buffer_text, buffer_chunk_ids, buffer_context))

        return merged_chunks