import logging
from typing import List, Dict, Any
from app.core.graph_db.neo4j.neo4j_session import neo4j_session


class Neo4jHandler:
    """
    Handles interactions with Neo4j for entity and relationship management,
    including storing and indexing embeddings for similarity search.
    """

    async def _get_session(self):
        """Helper to retrieve an active Neo4j session."""
        return await neo4j_session.get_session()

    async def _ensure_index_exists(self):
        """
        Ensures vector indexes exist in Neo4j. Creates them if missing.
        """
        async with await self._get_session() as session:
            # Check existing indexes
            check_query = """
            SHOW INDEXES YIELD name
            RETURN collect(name) AS indexes
            """
            result = await session.run(check_query)
            index_status = await result.single()
            existing_indexes = index_status["indexes"] if index_status else []

            entity_index_exists = "entity_embedding_index" in existing_indexes
            relation_index_exists = "relation_embedding_index" in existing_indexes

            # Create indexes if they don't exist
            if not entity_index_exists:
                logging.info("Creating entity embedding index in Neo4j...")
                await session.run("""
                CREATE VECTOR INDEX entity_embedding_index 
                FOR (e:Entity) ON (e.embedding)
                OPTIONS {
                    indexConfig: {
                        `vector.dimensions`: 256,
                        `vector.similarity_function`: 'cosine'
                    }
                };
                """)

            if not relation_index_exists:
                logging.info("Creating relationship embedding index in Neo4j...")
                await session.run("""
                CREATE VECTOR INDEX relation_embedding_index 
                FOR ()-[r:RELATION]->() ON (r.embedding)
                OPTIONS {
                    indexConfig: {
                        `vector.dimensions`: 256,
                        `vector.similarity_function`: 'cosine'
                    }
                };
                """)

    async def store_entities(self, entities: List[Dict[str, Any]], user_id: str, document_id: str):
        """
        Stores structured entities in Neo4j with automatic indexing.
        """
        await self._ensure_index_exists()  # Ensure indexes exist before insert

        query = """
        UNWIND $entities AS entity
        MERGE (e:Entity {id: entity.id, user_id: $user_id})
        ON CREATE SET 
            e.text = entity.text, 
            e.type = entity.entity_type, 
            e.profile = entity.entity_profile, 
            e.embedding = entity.embedding,
            e.document_id = $document_id
        RETURN elementId(e) AS neo4j_id
        """
        async with await self._get_session() as session:
            result = await session.run(query, entities=entities, user_id=user_id, document_id=document_id)
            neo4j_entities = await result.data()
            logging.info(f"Stored {len(entities)} entities with embeddings in Neo4j for user {user_id}.")
        return neo4j_entities

    async def store_relationships(self, relationships: List[Dict[str, Any]], user_id: str):
        """
        Stores structured relationships in Neo4j with automatic indexing.
        """
        await self._ensure_index_exists()  # Ensure indexes exist before insert

        query = """
        UNWIND $relationships AS rel
        MATCH (a:Entity {id: rel.source}), (b:Entity {id: rel.target})
        MERGE (a)-[r:RELATION {type: rel.relation_type, user_id: $user_id}]->(b)
        ON CREATE SET 
            r.confidence = coalesce(rel.confidence, 1.0),
            r.profile = rel.relation_profile,
            r.embedding = rel.embedding
        RETURN elementId(r) AS neo4j_id
        """
        async with await self._get_session() as session:
            result = await session.run(query, relationships=relationships, user_id=user_id)
            neo4j_relationships = await result.data()
            logging.info(f"Stored {len(relationships)} relationships with embeddings in Neo4j for user {user_id}.")
        return neo4j_relationships

    async def query_entities(self, entity_type: str, limit: int = 10):
        """
        Retrieves entities of a specific type with their profile and embeddings.
        """
        query = """
        MATCH (e:Entity)
        WHERE e.type = $entity_type
        RETURN e.id AS id, e.text AS text, e.type AS type, e.profile AS profile, e.embedding AS embedding
        LIMIT $limit
        """
        async with await self._get_session() as session:
            result = await session.run(query, entity_type=entity_type, limit=limit)
            return [record for record in await result.data()]

    async def query_relationships(self, source_id: str, relation: str, limit: int = 10):
        """
        Retrieves relationships of a specific type along with relation profiles and embeddings.
        """
        query = """
        MATCH (a:Entity {id: $source_id})-[r:RELATION]->(b:Entity)
        WHERE r.type = $relation
        RETURN b.id AS id, b.text AS text, b.type AS type, r.profile AS relation_profile, 
               r.confidence AS confidence, r.embedding AS embedding
        LIMIT $limit
        """
        async with await self._get_session() as session:
            result = await session.run(query, source_id=source_id, relation=relation, limit=limit)
            return [record for record in await result.data()]

    async def find_similar_entities(self, embedding: List[float], limit: int = 5):
        """
        Finds similar entities based on embedding similarity.
        """
        query = """
        CALL db.index.vector.queryNodes('entity_embedding_index', $limit, $embedding)
        YIELD node, score
        RETURN node.id AS id, node.text AS text, node.type AS type, node.profile AS profile, score
        ORDER BY score DESC
        LIMIT $limit
        """
        async with await self._get_session() as session:
            result = await session.run(query, embedding=embedding, limit=limit)
            return [record for record in await result.data()]

    async def find_similar_relationships(self, embedding: List[float], limit: int = 5):
        """
        Finds similar relationships based on embedding similarity.
        """
        query = """
        CALL db.index.vector.queryRelationships('relation_embedding_index', $limit, $embedding)
        YIELD relationship, score
        RETURN relationship.type AS relation_type, relationship.profile AS relation_profile, score
        ORDER BY score DESC
        LIMIT $limit
        """
        async with await self._get_session() as session:
            result = await session.run(query, embedding=embedding, limit=limit)
            return [record for record in await result.data()]

    async def update_entity_occurrences(self, entity_id: str, user_id: str):
        """
        Increments the occurrence count for an entity.
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

    async def clear_user_data(self, user_id: str) -> Dict[str, int]:
        """
        Deletes all entities and relationships associated with a user.
        """
        count_entity_query = "MATCH (e:Entity) WHERE e.user_id = $user_id RETURN count(e) AS entity_count"
        count_rel_query = "MATCH ()-[r:RELATION]->() WHERE r.user_id = $user_id RETURN count(r) AS rel_count"
        delete_entity_query = "MATCH (e:Entity) WHERE e.user_id = $user_id DETACH DELETE e"
        delete_rel_query = "MATCH ()-[r:RELATION]->() WHERE r.user_id = $user_id DELETE r"

        async with await self._get_session() as session:
            entity_result = await session.run(count_entity_query, user_id=user_id)
            entity_record = await entity_result.single()
            entities_count = entity_record["entity_count"] if entity_record else 0

            rel_result = await session.run(count_rel_query, user_id=user_id)
            rel_record = await rel_result.single()
            relationships_count = rel_record["rel_count"] if rel_record else 0

            await session.run(delete_entity_query, user_id=user_id)
            await session.run(delete_rel_query, user_id=user_id)

            logging.info(f"Deleted {entities_count} entities and {relationships_count} relationships for user {user_id}.")

            return {
                "entities_deleted": entities_count,
                "relationships_deleted": relationships_count
            }

# Instantiate a global Neo4j handler
neo4j_handler = Neo4jHandler()