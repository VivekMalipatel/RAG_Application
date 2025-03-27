import logging
from typing import List, Dict, Any
from app.core.graph_db.neo4j.neo4j_session import neo4j_session


class Neo4jSearchHandler:
    """
    Implements entity and relationship search within Neo4j-based knowledge graphs
    using structured retrieval methods based on LightRAG's research.
    """

    async def _get_session(self):
        """Helper method to obtain an active Neo4j session."""
        return await neo4j_session.get_session()

    async def search_entities(
        self, user_id: str, query_embedding: List[float], entity_type: str = None, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Searches for entities in Neo4j that match the given query embedding using vector similarity.

        Args:
            user_id (str): The unique identifier for the user performing the search.
            query_embedding (List[float]): The precomputed embedding of the query.
            entity_type (str, optional): Restricts search to a specific entity type.
            limit (int, optional): Number of results to return.

        Returns:
            List[Dict[str, Any]]: Matching entities with metadata.
        """
        try:
            async with await self._get_session() as session:
                query = """
                CALL db.index.vector.queryNodes('entity_embedding_index', $limit, $embedding)
                YIELD node, score
                WHERE node.user_id = $user_id
                AND ($entity_type IS NULL OR node.type = $entity_type)
                RETURN node.id AS id, node.text AS text, node.type AS type, node.profile AS profile, score
                ORDER BY score DESC
                LIMIT $limit
                """
                result = await session.run(
                    query, user_id=user_id, embedding=query_embedding, entity_type=entity_type, limit=limit
                )
                entities = await result.data()

            return entities

        except Exception as e:
            logging.error(f"Neo4j entity search failed: {str(e)}")
            return []

    async def search_relationships(
        self, user_id: str, query_embedding: List[float], relation_type: str = None, source_id: str = None, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Searches for relationships in Neo4j using vector similarity.

        Args:
            user_id (str): The unique identifier for the user performing the search.
            query_embedding (List[float]): The precomputed embedding of the query.
            relation_type (str, optional): Restrict search to a specific relation type.
            source_id (str, optional): Restrict search to relations from a specific entity.
            limit (int, optional): Number of results to return.

        Returns:
            List[Dict[str, Any]]: Matching relationships with metadata.
        """
        try:
            async with await self._get_session() as session:
                query = """
                CALL db.index.vector.queryRelationships('relation_embedding_index', $limit, $embedding)
                YIELD relationship, score
                WHERE startNode(relationship).user_id = $user_id
                AND ($relation_type IS NULL OR relationship.type = $relation_type)
                AND ($source_id IS NULL OR startNode(relationship).id = $source_id)
                RETURN startNode(relationship).id AS source, 
                       endNode(relationship).id AS target, 
                       relationship.type AS relation_type,
                       relationship.profile AS relation_profile,
                       score
                ORDER BY score DESC
                LIMIT $limit
                """
                result = await session.run(
                    query, user_id=user_id, embedding=query_embedding, relation_type=relation_type, source_id=source_id, limit=limit
                )
                relationships = await result.data()

            return relationships

        except Exception as e:
            logging.error(f"Neo4j relationship search failed: {str(e)}")
            return []

    async def retrieve_knowledge_paths(
        self, user_id: str, entity_id: str, max_depth: int = 3, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieves knowledge paths between entities in the graph based on entity relationships.

        Args:
            user_id (str): The unique identifier for the user performing the search.
            entity_id (str): The entity ID to start traversal from.
            max_depth (int, optional): The maximum traversal depth.
            limit (int, optional): Number of paths to return.

        Returns:
            List[Dict[str, Any]]: Paths consisting of entities and relationships.
        """
        try:
            async with await self._get_session() as session:
                query = f"""
                MATCH path = (start:Entity {{id: $entity_id, user_id: $user_id}})-[*1..{max_depth}]-(end:Entity)
                WITH path, relationships(path) AS rels, nodes(path) AS nodes
                RETURN path,
                       [n IN nodes | {{
                           id: n.id, 
                           text: n.text, 
                           type: n.type, 
                           profile: n.profile
                       }}] AS entities,
                       [r IN rels | {{
                           type: type(r),
                           relation_type: r.type,
                           profile: r.profile
                       }}] AS relations
                LIMIT $limit
                """
                result = await session.run(query, user_id=user_id, entity_id=entity_id, limit=limit)
                paths = await result.data()

            return paths

        except Exception as e:
            logging.error(f"Knowledge path retrieval failed: {str(e)}")
            return []