import logging
from app.core.graph_db.neo4j.neo4j_session import Neo4jSession
from typing import List, Dict, Any


class Neo4jHandler:
    """
    Handles interactions with Neo4j for entity and relationship management.
    """

    def __init__(self):
        """Initializes Neo4j session."""
        self.neo4j_session = Neo4jSession()
    
    async def _get_session(self):
        """Helper to retrieve an active Neo4j session."""
        return await self.neo4j_session.get_session()

    async def create_entity(self, entity: Dict[str, Any]):
        """
        Creates an entity node in Neo4j if it doesn't exist.
        """
        query = """
        MERGE (e:Entity {id: $entity_id})
        ON CREATE SET e.type = $entity_type, e += $properties
        RETURN e
        """
        async with await self._get_session() as session:
            await session.run(
                query,
                entity_id=entity.get("id"),
                entity_type=entity.get("label"),
                properties=entity
            )
            logging.info(f"Created or updated entity {entity.get('id')} in Neo4j.")

    async def create_relationship(self, relationship: Dict[str, Any], user_id: str):
        """
        Creates a relationship between two entities in Neo4j.
        """
        query = """
        MATCH (a:Entity {id: $source_id}), (b:Entity {id: $target_id})
        MERGE (a)-[r:RELATION {type: $relation, user_id: $user_id}]->(b)
        ON CREATE SET r.confidence = $confidence
        RETURN r
        """
        async with await self._get_session() as session:
            await session.run(
                query,
                source_id=relationship.get("source"),
                target_id=relationship.get("target"),
                relation=relationship.get("type"),
                confidence=relationship.get("confidence", 1.0),  # Default confidence to 1.0 if missing
                user_id=user_id
            )
            logging.info(
                f"Created relationship {relationship.get('type')} "
                f"between {relationship.get('source')} and {relationship.get('target')} in Neo4j."
            )

    async def query_entities(self, entity_type: str, limit: int = 10):
        """
        Retrieves entities of a specific type from Neo4j.
        Ensures the type property exists before querying.
        """
        query = """
        MATCH (e:Entity)
        WHERE e.type IS NOT NULL AND e.type = $entity_type
        RETURN coalesce(e.id, "") AS id, coalesce(e.text, "") AS text, coalesce(e.type, "") AS type
        LIMIT $limit
        """
        async with await self._get_session() as session:
            result = await session.run(query, entity_type=entity_type, limit=limit)
            return [record for record in await result.data()]

    async def query_relationships(self, source_id: str, relation: str, limit: int = 10):
        """
        Retrieves relationships of a specific type from Neo4j.
        """
        query = """
        MATCH (a:Entity {id: $source_id})-[r:RELATION]->(b:Entity)
        WHERE r.type IS NOT NULL AND r.type = $relation
        RETURN b.id AS id, coalesce(b.text, "") AS text, coalesce(b.type, "") AS type
        LIMIT $limit
        """
        async with await self._get_session() as session:
            result = await session.run(query, source_id=source_id, relation=relation, limit=limit)
            return [record for record in await result.data()]

    async def store_entities(self, entities: List[Dict[str, Any]], user_id: str, document_id: str):
        """
        Stores extracted entities in Neo4j.
        """
        query = """
        UNWIND $entities AS entity
        MERGE (e:Entity {id: entity.id})
        ON CREATE SET 
            e.text = entity.text, 
            e.type = entity.label, 
            e.user_id = $user_id, 
            e.document_id = $document_id
        RETURN e
        """
        async with await self._get_session() as session:
            await session.run(query, entities=entities, user_id=user_id, document_id=document_id)
            logging.info(f"Stored {len(entities)} entities in Neo4j for user {user_id}.")

    async def store_relationships(self, relationships: List[Dict[str, Any]], user_id: str):
        """
        Stores extracted relationships in Neo4j.
        """
        query = """
        UNWIND $relationships AS rel
        MATCH (a:Entity {id: rel.source}), (b:Entity {id: rel.target})
        MERGE (a)-[r:RELATION {type: rel.type, user_id: $user_id}]->(b)
        ON CREATE SET r.confidence = coalesce(rel.confidence, 1.0)
        RETURN r
        """
        async with await self._get_session() as session:
            await session.run(query, relationships=relationships, user_id=user_id)
            logging.info(f"Stored {len(relationships)} relationships in Neo4j for user {user_id}.")

    async def get_user_entities(self, user_id: str):
        """
        Retrieves stored entities for a user from Neo4j.
        Ensures that only entities with a `user_id` property are returned.
        """
        query = """
        MATCH (e:Entity)
        WHERE e.user_id IS NOT NULL AND e.user_id = $user_id
        RETURN coalesce(e.id, "") AS id, coalesce(e.text, "") AS text, coalesce(e.type, "") AS label
        """
        async with await self._get_session() as session:
            result = await session.run(query, user_id=user_id)
            records = await result.data()

            if not records:
                logging.info(f"No entities found for user: {user_id}")
                return []

            return records

    async def find_similar_entities(self, entity: Dict[str, Any], user_id: str):
        """
        Finds entities in Neo4j that share similar relationships & properties.
        Uses relationship-based similarity scoring.
        """
        query = """
        MATCH (e:Entity)-[r]-(related)
        WHERE e.text IS NOT NULL AND (e.text = $text OR e.type = $label)
        RETURN coalesce(e.id, "") AS id, coalesce(e.text, "") AS text, coalesce(e.type, "") AS label, COUNT(r) AS similarity_score
        ORDER BY similarity_score DESC
        LIMIT 5
        """
        async with await self._get_session() as session:
            result = await session.run(
                query,
                text=entity.get("text", ""),
                label=entity.get("label", "")
            )
            records = await result.data()

            if not records:
                return []

            return [
                {
                    "id": record["id"],
                    "text": record["text"],
                    "label": record["label"],
                    "similarity_score": record["similarity_score"]
                }
                for record in records
            ]

    async def update_entity_occurrences(self, entity_id: str, user_id: str):
        """
        Increments the occurrence count for an entity.
        If the entity doesn't exist, it does nothing.
        """
        query = """
        MATCH (e:Entity {id: $entity_id, user_id: $user_id})
        SET e.occurrences = coalesce(e.occurrences, 0) + 1
        RETURN e.occurrences
        """
        async with await self._get_session() as session:
            result = await session.run(query, entity_id=entity_id, user_id=user_id)
            record = await result.single()
            if record:
                logging.info(f"Updated occurrences for entity {entity_id} to {record['e.occurrences']}.")
            else:
                logging.warning(f"Entity {entity_id} not found for user {user_id}.")


# Instantiate a global Neo4j handler
neo4j_handler = Neo4jHandler()