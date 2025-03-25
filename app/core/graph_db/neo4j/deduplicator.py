import logging
import asyncio
from typing import List, Dict
from app.core.graph_db.neo4j.neo4j_session import Neo4jSession
from app.core.vector_store.qdrant.qdrant_handler import QdrantHandler
from app.core.graph_db.neo4j.neo4j_handler import Neo4jHandler
from app.config import settings


class Neo4jDeduplicator:
    """
    Handles deduplication of Neo4j graphs per user, merging duplicate entities
    and relationships while ensuring data integrity in Qdrant.
    """

    def __init__(self):
        self.qdrant = QdrantHandler()
        self.neo4j_session = Neo4jSession()
        self.neo4j_handler = Neo4jHandler()
    
    async def initialize(self):
        """
        Ensures vector indexes exist before deduplication starts.
        """
        logging.info("Checking Neo4j vector indexes before starting deduplication...")
        await self.neo4j_handler._ensure_index_exists()  # Ensure indexes exist

    async def find_duplicate_entities(self, user_id: int) -> List[Dict[str, str]]:
        """
        Identifies duplicate entities based on vector similarity and text similarity.
        Returns a list of duplicate entity ID pairs.
        """
        try:
            async with await self.neo4j_session.get_session() as session:
                query = """
                MATCH (n:Entity {user_id: $user_id})
                CALL db.index.vector.queryNodes('entity_embedding_index', 1009, n.embedding)
                YIELD node AS similarNode, score
                WHERE score > 0.85 AND n <> similarNode
                AND apoc.text.levenshteinSimilarity(n.text, similarNode.text) > 0.8
                AND n.type = similarNode.type
                RETURN n.id AS original_id, similarNode.id AS duplicate_id, score;
                """

                # Execute query with parameterized input
                result = await session.run(query, user_id=user_id)
                duplicates = await result.data()

                if not duplicates:
                    logging.warning(f"No duplicates found for user {user_id}.")
                else:
                    logging.info(f"Found {len(duplicates)} duplicates for user {user_id}")

                return duplicates

        except Exception as e:
            logging.error(f"Error finding duplicate entities for user {user_id}: {str(e)}")
            logging.error(f"Exception type: {type(e)}")
            return []

    async def merge_entities(self, user_id: int, keep_id: str, remove_id: str):
        """
        Merges a single pair of duplicate entities in Neo4j and updates Qdrant.
        """
        user_id = int(user_id)
        try:
            async with await self.neo4j_session.get_session() as session:
                # Merge entities: Keep the one selected
                merge_query = """
                MATCH (keep:Entity {id: $keep_id, user_id: $user_id}),
                      (remove:Entity {id: $remove_id, user_id: $user_id})
                CALL apoc.refactor.mergeNodes([keep, remove], {properties: ["overwrite"]}) YIELD node
                RETURN elementId(node) AS merged_id;
                """
                merged_result = await session.run(merge_query, keep_id=keep_id, remove_id=remove_id, user_id=user_id)
                merged_node = await merged_result.single()

                if merged_node:
                    new_id = merged_node["merged_id"]

                    logging.info(f"Merged entity {remove_id} → {new_id} (keeping {keep_id}) for user {user_id}")

        except Exception as e:
            logging.error(f"Error merging entities for user {user_id}: {str(e)}")
    
    async def _select_best_entity(self, user_id: int, duplicate_entities, entity_count):
        """
        Determines the best entity to keep based on occurrence frequency, breaking ties with text length.

        Returns:
            (keep_id, remove_id): The entity to keep and the entity to remove.
        """

        async with await self.neo4j_session.get_session() as session:
            # Fetch text lengths for entities
            entity_texts = {}
            for pair in duplicate_entities:
                entity_texts[pair["original_id"]] = None
                entity_texts[pair["duplicate_id"]] = None

            # Get text lengths from Neo4j
            query = """
            UNWIND $entity_ids AS id
            MATCH (e:Entity {user_id: $user_id, id: id})
            RETURN e.id AS entity_id, size(e.text) AS text_length;
            """
            result = await session.run(query, user_id=user_id, entity_ids=list(entity_texts.keys()))
            text_lengths = await result.data()
            for record in text_lengths:
                entity_texts[record["entity_id"]] = record["text_length"]

            # Determine the best entity to keep based on frequency and text length
            best_pair = None
            for pair in duplicate_entities:
                original_id, duplicate_id = pair["original_id"], pair["duplicate_id"]

                # Get occurrence frequency
                count_original = entity_count.get(original_id, 0)
                count_duplicate = entity_count.get(duplicate_id, 0)

                if count_original > count_duplicate:
                    best_pair = (original_id, duplicate_id)
                elif count_original < count_duplicate:
                    best_pair = (duplicate_id, original_id)
                else:  # Tie-breaker: Use text length
                    if entity_texts[original_id] >= entity_texts[duplicate_id]:
                        best_pair = (original_id, duplicate_id)
                    else:
                        best_pair = (duplicate_id, original_id)

            return best_pair

    async def deduplicate_user_graph(self, user_id: int):
        """
        Runs full deduplication (entities & relationships) for a user's graph.
        Continuously fetches duplicates and merges until none exist.
        """
        try:
            while True:
                duplicate_entities = await self.find_duplicate_entities(user_id)

                if not duplicate_entities:
                    logging.info(f"No more duplicates for user {user_id}. Deduplication complete.")
                    break  # Exit when no duplicates are found

                # Count occurrences of each entity in duplicates
                entity_count = {}
                for pair in duplicate_entities:
                    entity_count[pair["original_id"]] = entity_count.get(pair["original_id"], 0) + 1
                    entity_count[pair["duplicate_id"]] = entity_count.get(pair["duplicate_id"], 0) + 1

                # Select the most frequent entity to keep, break ties by text length
                entity_to_keep, entity_to_remove = await self._select_best_entity(user_id, duplicate_entities, entity_count)

                # Merge one duplicate pair at a time
                await self.merge_entities(user_id, entity_to_keep, entity_to_remove)

        except Exception as e:
            logging.error(f"Deduplication failed for user {user_id}: {str(e)}")

    async def start_deduplication(self):
        """
        Runs deduplication for all users at fixed intervals.
        """
        await self.initialize()
        while True:
            users = await self.qdrant.get_all_containers()
            await asyncio.gather(*(self.deduplicate_user_graph(int(user_id)) for user_id in users))
            await asyncio.sleep(settings.NEO4J_DEDUPLICATION_INTERVAL)