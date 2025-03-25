import uuid
import logging
import json
from datetime import datetime
from typing import List, Dict, Any, Tuple
from pydantic import BaseModel
from app.services.file_processor.entity_relation_extractor import EntityRelationExtractor, EntityRelationSchema
from app.core.graph_db.neo4j.neo4j_handler import Neo4jHandler
from app.core.vector_store.qdrant.qdrant_handler import QdrantHandler
from app.core.embedding.embedding_handler import EmbeddingHandler
from app.api.models import Chat
from app.api.core.crud import CRUDBase
from app.api.db import AsyncSessionLocal
from sqlalchemy.future import select
from app.core.models.model_handler import ModelRouter
from app.core.models.model_type import ModelType
from app.core.models.model_provider import Provider
from app.config import settings

class ChatSummarySchema(BaseModel):
    summary: str

class ChatHandler:
    """Abstract Chat Processing Class for handling long-term memory of chat messages."""

    def __init__(self):
        """Initialize dependencies for chat processing."""
        self.qdrant = QdrantHandler()
        self.entity_relation_extractor = EntityRelationExtractor()
        self.embedding_handler = EmbeddingHandler()
        self.neo4j = Neo4jHandler()
        running_summary_system_prompt = """
            You are an AI conversation analyst that creates concise summaries of ongoing chats.
            
            Your task:
            1. Review the current chat summary (if available)
            2. Analyze the new message content
            3. Generate an updated summary that captures the key points and context of the conversation
            
            Focus on maintaining continuity while incorporating new information.
            Highlight important entities, facts, topics, and user intentions.
            Keep your summary concise, factual, and useful for chat context retrieval.
            
            Respond with a structured JSON containing only the summary:
            {
            "summary": "<concise chat summary text>"
            }
            """
        self.chat_summary_model = ModelRouter(
                provider=Provider.OLLAMA,  
                model_name="meta-llama/Llama-3.1-8B-Instruct",
                model_quantization="Q8_0",
                model_type=ModelType.TEXT_GENERATION,
                system_prompt=running_summary_system_prompt,
                temperature=settings.TEXT_CONTEXT_LLM_TEMPERATURE,
                top_p=settings.TEXT_CONTEXT_LLM_TOP_P,
                max_tokens=settings.TEXT_CHUNK_CONTEXT_MAX_TOKENS,
            )

    async def process_chat(self, user_id: str, message: str, message_type: str):
        """
        Processes a chat message by extracting entities, storing in DB, and embedding for retrieval.

        Steps:
        1. Extract entities and relationships.
        2. Generate embeddings for entities and relationships separately.
        3. Add extracted information to chat metadata.
        4. Store chat message in PostgreSQL.
        5. Generate dense & sparse embeddings for the full payload.
        6. Store embeddings in Qdrant.
        """

        chat_id = str(uuid.uuid4())
        timestamp = datetime.utcnow()

        # Step 1: Extract Entities & Relationships
        entities, relations = await self.extract_entities_and_relations(message)

        chat_summary = await self.generate_running_chat_summary(chat_id, user_id, message, message_type, timestamp, entities, relations)

        # Step 4: Store chat message in PostgreSQL
        await self.store_chat_message(chat_id=chat_id, user_id=user_id, message=message, message_type=message_type, timestamp=timestamp, entities=entities, relationships=relations, chat_summary=chat_summary)

        message_payload = {
            "chat_id": chat_id,
            "user_id": user_id,
            "message": message,
            "message_type": message_type,
            "timestamp": timestamp,
            "entities": entities,
            "relationships": relations,
            "chat_summary": chat_summary
        }

        # Step 5: Generate dense & sparse embeddings for full message payload
        dense_embedding, sparse_embedding = await self.generate_embeddings(json.dumps(message_payload))

        # Step 6: Store in Qdrant
        await self.store_in_qdrant(chat_id=chat_id, user_id=user_id, message=message, message_type=message_type, timestamp=timestamp, entities=entities, relationships=relations, chat_summary=chat_summary, dense_embedding=dense_embedding, sparse_embedding=sparse_embedding)

        logging.info(f"Processed chat message for user {user_id}: {message}")

    async def extract_entities_and_relations(self, text: str, chat_summary: str, user_id:int, chat_id:str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Extracts entities and relationships from the given text.
        Returns:
            - List of extracted entities
            - List of extracted relationships
        """
        
        extraction_prompt = f"""
            Ensure that:
            - Entities are **correctly labeled** (PERSON, ORG, LOCATION, PRODUCT, etc.).
            - Each entity and relationship has a **profile description** explaining its significance.

            Extract structured knowledge graph entities & relationships from the following text:
            
            <START TEXT>
            {text}
            <END TEXT>


            For your context regarding the message this text is extracted from, please refer to the below chat summary (Just for your context, donot extract entities from this summary until and unless you are not able to get a meaningful entities and realtions from the text and its context):
            <START TEXT>
            {chat_summary}
            <END TEXT>

            """
        extracted_data: EntityRelationSchema = await self.entity_relation_extractor.llm.client.generate_structured_output(
                extraction_prompt, schema=EntityRelationSchema
            )
        entities = extracted_data.entities
        relationships = extracted_data.relationships
        
        if entities:
            entities_data, entity_texts = [], [], []
            for e in entities:
                entity_text = f"{e.text} {e.entity_type} {e.entity_profile}"
                entities_data.append({
                    "id": e.id,
                    "text": e.text,
                    "entity_type": e.entity_type,
                    "entity_profile": e.entity_profile
                })
                entity_texts.append(entity_text)

            # Generate embeddings for entities
            entity_embeddings = await self.entity_relation_extractor.embedding_handler.encode_dense(entity_texts)
            entity_embeddings = [emb[:256] if len(emb) >= 256 else emb for emb in entity_embeddings]

            # Attach embeddings to entities
            for i, entity in enumerate(entities_data):
                entity["embedding"] = entity_embeddings[i]

            # Store entities in Neo4j
            _ = await self.entity_relation_extractor.graph.store_entities(entities_data, user_id, chat_id)

        else:
            entities_data = []

        # Process relationships separately
        if relationships:
            relationships_data, relation_texts = [], [], []
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

            # Generate embeddings for relationships
            relation_embeddings = await self.entity_relation_extractor.embedding_handler.encode_dense(relation_texts)
            relation_embeddings = [emb[:256] if len(emb) >= 256 else emb for emb in relation_embeddings]

            # Attach embeddings to relationships
            for i, relation in enumerate(relationships_data):
                relation["embedding"] = relation_embeddings[i]

            # Store relationships in Neo4j
            _ = await self.entity_relation_extractor.graph.store_relationships(relationships_data, user_id)

        else:
            relationships_data = []

        # Remove "embedding" from entities_data and relationships_data
        for entity in entities_data:
            if "embedding" in entity:
                del entity["embedding"]

        for relationship in relationships_data:
            if "embedding" in relationship:
                del relationship["embedding"]

        return entities_data if entities_data else [], relationships_data if relationships_data else []
    
    async def generate_running_chat_summary(self, chat_id: str, user_id: str, message: str, message_type: str, timestamp: datetime, entities: List, relationships: List):
        """Generates a running chat summary for the user."""

        current_chat_summary = await self.get_current_chat_summary(chat_id)
        message_payload = {
            "chat_id": chat_id,
            "user_id": user_id,
            "message": message,
            "message_type": message_type,
            "timestamp": timestamp,
            "entities": entities,
            "relationships": relationships
        }
        chat_summary: ChatSummarySchema = await self.chat_summary_model.client.generate_structured_output(
            prompt=f"""

            Here is the chat summary for this chat until now:
            <START CHAT SUMMARY>
            {current_chat_summary if current_chat_summary else "No chat summary available until now."}
            <END CHAT SUMMARY>

            Here is the new message content:
            <START MESSAGE>
            {json.dumps(message_payload)}
            <END MESSAGE>
            """,
            schema=ChatSummarySchema,

        )
        return chat_summary.summary

    async def store_chat_message(
        self, chat_id: str, user_id: str, message: str, message_type: str, timestamp: datetime, entities: List, relationships: List, chat_summary: str
    ):
        """Stores chat messages in PostgreSQL using CRUD operations."""
        async with AsyncSessionLocal() as session:
            chat_data = {
                "chat_id": chat_id,
                "user_id": user_id,
                "timestamp": timestamp,
                "message": message,
                "message_type": message_type,
                "entities": entities,
                "relationships": relationships,
                "chat_summary": chat_summary
            }
            new_chat = await CRUDBase(Chat).create(db=session, obj_in=chat_data)
            return new_chat
    
    async def get_current_chat_summary(self, chat_id: str):
        """Fetches the latest chat summary for the given chat ID."""
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(Chat).filter_by(chat_id=chat_id).order_by(Chat.timestamp.desc()).limit(1)
            )
            chat_summary = result.scalar_one_or_none()
            return chat_summary

    async def generate_embeddings(self, text: str):
        """Generates dense & sparse embeddings for the entire chat payload."""
        dense_embedding = await self.embedding_handler.encode_dense(text)
        sparse_embedding = await self.embedding_handler.encode_sparse(text)
        return dense_embedding, sparse_embedding

    async def store_in_qdrant(
        self, chat_id: str, user_id: str, message: str, message_type: str, timestamp: datetime, entities: List, relationships: List, chat_summary: str, dense_embedding: List, sparse_embedding: Dict
    ):
        """Stores chat message embeddings in Qdrant."""
        await self.qdrant.store_chat_vectors(
            embedded_payload=[
                {
                    "chat_id": chat_id,
                    "message": message,
                    "message_type": message_type,
                    "timestamp": timestamp,
                    "entities": entities,
                    "relationships": relationships,
                    "chat_summary": chat_summary,
                    "dense_embedding": dense_embedding,
                    "sparse_embedding": sparse_embedding
                }
            ],
            user_id=user_id
        )
        logging.info(f"Chat message stored in Qdrant: {chat_id}")