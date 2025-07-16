
import logging
import asyncio
from neo4j import AsyncGraphDatabase, basic_auth
from typing import Dict, Any, List, Optional
import json
from config import settings
logger = logging.getLogger(__name__)

class Neo4jHandler:
    def __init__(self):
        self.uri = settings.NEO4J_URI
        self.username = settings.NEO4J_USERNAME
        self.password = settings.NEO4J_PASSWORD
        self.database = getattr(settings, 'NEO4J_DATABASE', 'neo4j')
        self.driver = None
        self._initialize_driver()

    def _initialize_driver(self):
        try:
            self.driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=basic_auth(self.username, self.password),
                max_connection_lifetime=30 * 60,
                max_connection_pool_size=50,
                connection_acquisition_timeout=30,
                connection_timeout=30
            )
            logger.info("Neo4j driver initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Neo4j driver: {e}")
            raise

    async def close(self):
        if self.driver:
            await self.driver.close()
            logger.info("Neo4j driver closed")

    async def test_connection(self) -> bool:
        try:
            async with self.driver.session(database=self.database) as session:
                result = await session.run("RETURN 1 as test")
                record = await result.single()
                return record["test"] == 1
        except Exception as e:
            logger.error(f"Neo4j connection test failed: {e}")
            return False

    async def create_indexes(self):
        try:
            indexes = [
                f"CREATE VECTOR INDEX page_embedding_index IF NOT EXISTS FOR (p:Page) ON (p.embedding) OPTIONS {{indexConfig: {{`vector.dimensions`: {settings.EMBEDDING_DIMENSIONS}, `vector.similarity_function`: 'cosine'}}}}",
                f"CREATE VECTOR INDEX entity_embedding_index IF NOT EXISTS FOR (e:Entity) ON (e.embedding) OPTIONS {{indexConfig: {{`vector.dimensions`: {settings.EMBEDDING_DIMENSIONS}, `vector.similarity_function`: 'cosine'}}}}",
                f"CREATE VECTOR INDEX column_embedding_index IF NOT EXISTS FOR (c:Column) ON (c.embedding) OPTIONS {{indexConfig: {{`vector.dimensions`: {settings.EMBEDDING_DIMENSIONS}, `vector.similarity_function`: 'cosine'}}}}",
                f"CREATE VECTOR INDEX relationship_embedding_index IF NOT EXISTS FOR ()-[r:RELATIONSHIP]->() ON (r.embedding) OPTIONS {{indexConfig: {{`vector.dimensions`: {settings.EMBEDDING_DIMENSIONS}, `vector.similarity_function`: 'cosine'}}}}",
                "CREATE INDEX document_user_org_index IF NOT EXISTS FOR (d:Document) ON (d.user_id, d.org_id)",
                "CREATE INDEX page_user_org_index IF NOT EXISTS FOR (p:Page) ON (p.user_id, p.org_id)",
                "CREATE INDEX entity_id_index IF NOT EXISTS FOR (e:Entity) ON (e.id)",
                "CREATE INDEX entity_type_index IF NOT EXISTS FOR (e:Entity) ON (e.entity_type)",
            ]
            
            async def create_single_index(index_query):
                try:
                    async with self.driver.session(database=self.database) as session:
                        await session.run(index_query)
                        logger.info(f"Created/verified index: {index_query.split(' ')[2]}")
                except Exception as e:
                    logger.warning(f"Index creation warning: {e}")
            
            await asyncio.gather(*[create_single_index(index_query) for index_query in indexes])
            logger.info("All indexes created/verified successfully")
        except Exception as e:
            logger.error(f"Error creating indexes: {e}")
            raise
    
    async def store_unstructured_document(self, document_data: Dict[str, Any]) -> bool:
        try:
            async with self.driver.session(database=self.database) as session:
                tx = await session.begin_transaction()
                try:
                    doc_properties = {
                        "user_id": document_data["user_id"],
                        "org_id": document_data["org_id"],
                        "s3_url": document_data["s3_url"],
                        "source": document_data["source"],
                        "filename": document_data["filename"],
                        "file_type": document_data["file_type"],
                        "category": document_data["category"],
                        "internal_object_id": document_data["internal_object_id"],
                        "task_id": document_data["task_id"]
                    }
                    
                    metadata = document_data.get("metadata", {})
                    for key, value in metadata.items():
                        if key not in doc_properties:
                            doc_properties[f"{key}"] = value
                    
                    doc_query = """
                    MERGE (d:Document {internal_object_id: $internal_object_id})
                    ON CREATE SET d = $doc_properties
                    ON MATCH SET d = $doc_properties
                    WITH d
                    OPTIONAL MATCH (d)-[:HAS_PAGE]->(p:Page)
                    OPTIONAL MATCH (p)-[:MENTIONS]->(e:Entity)
                    OPTIONAL MATCH (e)-[r:RELATIONSHIP]->(e2:Entity)
                    DETACH DELETE p, e, r
                    RETURN d
                    """
                    doc_result = await tx.run(doc_query, 
                                            internal_object_id=document_data["internal_object_id"],
                                            doc_properties=doc_properties)
                    document_node = await doc_result.single()
                    
                    async def process_page_data(page_data):
                        page_properties = {
                            "page_number": page_data["page_number"],
                            "user_id": document_data["user_id"],
                            "org_id": document_data["org_id"],
                            "image_s3_url": page_data["image_s3_url"],
                            "content": json.dumps(page_data["messages"][0]["content"])
                        }
                        
                        if "embedding" in page_data:
                            page_properties["embedding"] = page_data["embedding"]
                        
                        page_query = """
                        MATCH (d:Document {internal_object_id: $internal_object_id})
                        CREATE (p:Page $page_properties)
                        CREATE (d)-[:HAS_PAGE]->(p)
                        RETURN p
                        """
                        page_result = await tx.run(page_query, 
                                                 internal_object_id=document_data["internal_object_id"],
                                                 page_properties=page_properties)
                        page_node = await page_result.single()

                        for entity_data in page_data.get("entities", []):
                            entity_properties = {
                                "id": entity_data["id"],
                                "text": entity_data["text"],
                                "entity_type": entity_data["entity_type"],
                                "entity_profile": entity_data["entity_profile"]
                            }
                            
                            if "embedding" in entity_data:
                                entity_properties["embedding"] = entity_data["embedding"]
                            
                            entity_query = """
                            MERGE (e:Entity {id: $entity_id})
                            ON CREATE SET e = $entity_properties
                            ON MATCH SET e += $entity_properties
                            WITH e
                            MATCH (p:Page {page_number: $page_number, user_id: $user_id, org_id: $org_id})
                            MERGE (p)-[:MENTIONS]->(e)
                            RETURN e
                            """
                            await tx.run(entity_query,
                                       entity_id=entity_data["id"],
                                       entity_properties=entity_properties,
                                       page_number=page_data["page_number"],
                                       user_id=document_data["user_id"],
                                       org_id=document_data["org_id"])

                        for rel_data in page_data.get("relationships", []):
                            rel_properties = {
                                "relation_type": rel_data["relation_type"],
                                "relation_profile": rel_data["relation_profile"]
                            }
                            
                            if "embedding" in rel_data:
                                rel_properties["embedding"] = rel_data["embedding"]
                            
                            rel_query = """
                            MATCH (source:Entity {id: $source_id})
                            MATCH (target:Entity {id: $target_id})
                            MERGE (source)-[r:RELATIONSHIP]->(target)
                            SET r += $rel_properties
                            RETURN r
                            """
                            await tx.run(rel_query,
                                       source_id=rel_data["source"],
                                       target_id=rel_data["target"],
                                       rel_properties=rel_properties)

                    for page_data in document_data.get("data", []):
                        await process_page_data(page_data)
                    
                    await tx.commit()
                    logger.info(f"Successfully stored unstructured document: {document_data['filename']}")
                    return True
                except Exception as e:
                    await tx.rollback()
                    raise e
                    
        except Exception as e:
            logger.error(f"Error storing unstructured document: {e}")
            return False
    
    async def store_structured_document(self, document_data: Dict[str, Any]) -> bool:
        try:
            async with self.driver.session(database=self.database) as session:
                tx = await session.begin_transaction()
                try:
                    doc_properties = {
                        "user_id": document_data["user_id"],
                        "org_id": document_data["org_id"],
                        "s3_url": document_data["s3_url"],
                        "source": document_data["source"],
                        "filename": document_data["filename"],
                        "file_type": document_data["file_type"],
                        "category": document_data["category"],
                        "internal_object_id": document_data["internal_object_id"],
                        "task_id": document_data["task_id"]
                    }
                    
                    metadata = document_data.get("metadata", {})
                    for key, value in metadata.items():
                        doc_properties[f"metadata_{key}"] = value
                    
                    doc_query = """
                    MERGE (d:Document {internal_object_id: $internal_object_id})
                    ON CREATE SET d = $doc_properties
                    ON MATCH SET d = $doc_properties
                    WITH d
                    OPTIONAL MATCH (d)-[:HAS_PAGE]->(p:Page)
                    OPTIONAL MATCH (p)-[:MENTIONS]->(c:Column)
                    OPTIONAL MATCH (p)-[:MENTIONS]->(e:Entity)
                    OPTIONAL MATCH (e)-[r_rel:RELATIONSHIP]->(e2:Entity)
                    DETACH DELETE p, c, e, r_rel
                    RETURN d
                    """
                    await tx.run(doc_query, 
                                internal_object_id=document_data["internal_object_id"],
                                doc_properties=doc_properties)
                    
                    async def process_sheet_data(sheet_data):
                        if sheet_data.get("is_tabular", False):
                            page_properties = {
                                "page_number": 1,
                                "user_id": document_data["user_id"],
                                "org_id": document_data["org_id"],
                                "sheet_name": sheet_data["sheet_name"],
                                "summary": sheet_data["summary"],
                                "total_rows": sheet_data["total_rows"],
                                "total_columns": sheet_data["total_columns"],
                                "is_tabular": True
                            }
                            
                            if "summary_embedding" in sheet_data:
                                page_properties["embedding"] = sheet_data["summary_embedding"]
                            
                            page_query = """
                            MATCH (d:Document {internal_object_id: $internal_object_id})
                            CREATE (p:Page $page_properties)
                            CREATE (d)-[:HAS_PAGE]->(p)
                            RETURN p
                            """
                            await tx.run(page_query,
                                       internal_object_id=document_data["internal_object_id"],
                                       page_properties=page_properties)

                            for col_data in sheet_data.get("column_profiles", []):
                                col_properties = {
                                    "column_name": col_data["column_name"],
                                    "column_profile": col_data["column_profile"]
                                }
                                
                                if "embedding" in col_data:
                                    col_properties["embedding"] = col_data["embedding"]
                                
                                col_query = """
                                MATCH (p:Page {sheet_name: $sheet_name, user_id: $user_id, org_id: $org_id})
                                CREATE (c:Column $col_properties)
                                CREATE (p)-[:MENTIONS]->(c)
                                RETURN c
                                """
                                await tx.run(col_query,
                                           sheet_name=sheet_data["sheet_name"],
                                           user_id=document_data["user_id"],
                                           org_id=document_data["org_id"],
                                           col_properties=col_properties)
                        else:
                            for chunk_data in sheet_data.get("chunks", []):
                                page_properties = {
                                    "page_number": chunk_data["page_number"],
                                    "user_id": document_data["user_id"],
                                    "org_id": document_data["org_id"],
                                    "text": chunk_data["text"],
                                    "sheet_name": sheet_data["sheet_name"],
                                    "is_tabular": False
                                }
                                
                                page_query = """
                                MATCH (d:Document {internal_object_id: $internal_object_id})
                                CREATE (p:Page $page_properties)
                                CREATE (d)-[:HAS_PAGE]->(p)
                                RETURN p
                                """
                                await tx.run(page_query,
                                           internal_object_id=document_data["internal_object_id"],
                                           page_properties=page_properties)

                                await self._process_entities_relationships(tx, chunk_data, document_data)

                    for sheet_data in document_data.get("data", []):
                        await process_sheet_data(sheet_data)
                    
                    await tx.commit()
                    logger.info(f"Successfully stored structured document: {document_data['filename']}")
                    return True
                except Exception as e:
                    await tx.rollback()
                    raise e
                    
        except Exception as e:
            logger.error(f"Error storing structured document: {e}")
            return False
    
    async def store_direct_document(self, document_data: Dict[str, Any]) -> bool:
        try:
            async with self.driver.session(database=self.database) as session:
                tx = await session.begin_transaction()
                try:
                    doc_properties = {
                        "user_id": document_data["user_id"],
                        "org_id": document_data["org_id"],
                        "s3_url": document_data["s3_url"],
                        "source": document_data["source"],
                        "filename": document_data["filename"],
                        "file_type": document_data["file_type"],
                        "category": document_data["category"],
                        "internal_object_id": document_data["internal_object_id"],
                        "task_id": document_data["task_id"]
                    }

                    metadata = document_data.get("metadata", {})
                    for key, value in metadata.items():
                        doc_properties[f"metadata_{key}"] = value
                    
                    doc_query = """
                    MERGE (d:Document {internal_object_id: $internal_object_id})
                    ON CREATE SET d = $doc_properties
                    ON MATCH SET d = $doc_properties
                    WITH d
                    OPTIONAL MATCH (d)-[:HAS_PAGE]->(p:Page)
                    OPTIONAL MATCH (p)-[:MENTIONS]->(e:Entity)
                    OPTIONAL MATCH (e)-[r:RELATIONSHIP]->(e2:Entity)
                    DETACH DELETE p, e, c, r
                    RETURN d
                    """
                    await tx.run(doc_query, 
                                internal_object_id=document_data["internal_object_id"],
                                doc_properties=doc_properties)
                    
                    async def process_chunk_data(chunk_data):
                        page_properties = {
                            "page_number": chunk_data["page_number"],
                            "user_id": document_data["user_id"],
                            "org_id": document_data["org_id"],
                            "text": chunk_data["text"]
                        }
                        
                        page_query = """
                        MATCH (d:Document {internal_object_id: $internal_object_id})
                        CREATE (p:Page $page_properties)
                        CREATE (d)-[:HAS_PAGE]->(p)
                        RETURN p
                        """
                        await tx.run(page_query,
                                   internal_object_id=document_data["internal_object_id"],
                                   page_properties=page_properties)
                        
                        await self._process_entities_relationships(tx, chunk_data, document_data)

                    for chunk_data in document_data.get("data", []):
                        await process_chunk_data(chunk_data)
                    
                    await tx.commit()
                    logger.info(f"Successfully stored direct document: {document_data['filename']}")
                    return True
                except Exception as e:
                    await tx.rollback()
                    raise e
                    
        except Exception as e:
            logger.error(f"Error storing direct document: {e}")
            return False
    
    async def _process_entities_relationships(self, tx, chunk_data: Dict[str, Any], document_data: Dict[str, Any]):
        for entity_data in chunk_data.get("entities", []):
            entity_properties = {
                "id": entity_data["id"],
                "text": entity_data["text"],
                "entity_type": entity_data["entity_type"],
                "entity_profile": entity_data["entity_profile"]
            }
            
            if "embedding" in entity_data:
                entity_properties["embedding"] = entity_data["embedding"]
            
            entity_query = """
            MERGE (e:Entity {id: $entity_id})
            ON CREATE SET e = $entity_properties
            ON MATCH SET e += $entity_properties
            WITH e
            MATCH (p:Page {page_number: $page_number, user_id: $user_id, org_id: $org_id})
            MERGE (p)-[:MENTIONS]->(e)
            RETURN e
            """
            await tx.run(entity_query,
                       entity_id=entity_data["id"],
                       entity_properties=entity_properties,
                       page_number=chunk_data["page_number"],
                       user_id=document_data["user_id"],
                       org_id=document_data["org_id"])

        for rel_data in chunk_data.get("relationships", []):
            rel_properties = {
                "relation_type": rel_data["relation_type"],
                "confidence": rel_data["confidence"],
                "relation_profile": rel_data["relation_profile"]
            }
            
            if "embedding" in rel_data:
                rel_properties["embedding"] = rel_data["embedding"]
            
            rel_query = """
            MATCH (source:Entity {id: $source_id})
            MATCH (target:Entity {id: $target_id})
            MERGE (source)-[r:RELATIONSHIP]->(target)
            SET r += $rel_properties
            RETURN r
            """
            await tx.run(rel_query,
                       source_id=rel_data["source"],
                       target_id=rel_data["target"],
                       rel_properties=rel_properties)
    
    async def execute_cypher_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        try:
            async with self.driver.session(database=self.database) as session:
                result = await session.run(query, parameters or {})
                records = await result.data()
                return [dict(record) for record in records]
                
        except Exception as e:
            logger.error(f"Error executing Cypher query: {e}")
            return []
    
    async def delete_document(self, user_id: str, org_id: str, internal_object_id: str) -> bool:
        try:
            async with self.driver.session(database=self.database) as session:
                query = """
                MATCH (d:Document {user_id: $user_id, org_id: $org_id, internal_object_id: $internal_object_id})
                OPTIONAL MATCH (d)-[:HAS_PAGE]->(p:Page)
                OPTIONAL MATCH (p)-[:MENTIONS]->(e:Entity)
                OPTIONAL MATCH (p)-[:MENTIONS]->(c:Column)
                OPTIONAL MATCH (p)-[:HAS_COLUMN]->(c2:Column)
                OPTIONAL MATCH (e)-[r:RELATIONSHIP]->(e2:Entity)
                DETACH DELETE d, p, e, c, c2, r, e2
                """
                
                await session.run(query, 
                                user_id=user_id,
                                org_id=org_id,
                                internal_object_id=internal_object_id)
                
                logger.info(f"Deleted document: {internal_object_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            return False

_neo4j_handler: Optional[Neo4jHandler] = None

def get_neo4j_handler() -> Neo4jHandler:
    global _neo4j_handler
    if _neo4j_handler is None:
        _neo4j_handler = Neo4jHandler()
    return _neo4j_handler

async def initialize_neo4j():
    handler = get_neo4j_handler()
    if await handler.test_connection():
        await handler.create_indexes()
        logger.info("Neo4j initialized successfully")
    else:
        raise Exception("Failed to connect to Neo4j")

async def cleanup_neo4j():
    global _neo4j_handler
    if _neo4j_handler:
        await _neo4j_handler.close()
        _neo4j_handler = None
