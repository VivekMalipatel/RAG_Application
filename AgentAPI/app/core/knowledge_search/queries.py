import aiohttp
import logging
from typing import Optional, Dict, Any, List
from embed.embed import JinaEmbeddings
from config import config as app_config

logger = logging.getLogger(__name__)

_embeddings_client = None

def _get_embeddings_client() -> JinaEmbeddings:
    global _embeddings_client
    if _embeddings_client is None:
        _embeddings_client = JinaEmbeddings()
    return _embeddings_client

def _prune_embeddings(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _prune_embeddings(item)
            for key, item in value.items()
            if "embedding" not in key.lower()
        }
    if isinstance(value, list):
        return [_prune_embeddings(item) for item in value]
    return value

async def _generate_embedding(text: str) -> List[float]:
    embeddings_client = _get_embeddings_client()
    try:
        embedding = await embeddings_client.aembed_query(text)
        return embedding
    except Exception as e:
        logger.error(f"embedding_generation_failed: {str(e)}")
        raise ValueError(f"Failed to generate embedding: {str(e)}")

async def _execute_cypher(
    cypher_query: str,
    parameters: Dict[str, Any],
    user_id: str,
    org_id: str
) -> List[Dict[str, Any]]:
    url = f"{app_config.INDEXER_API_BASE_URL}/search/cypher"
    
    parameters["user_id"] = str(user_id).split('$')[0]
    parameters["org_id"] = str(org_id).split('$')[0]
    
    payload = {
        "query": cypher_query,
        "parameters": parameters
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    return _prune_embeddings(result)
                else:
                    error_text = await response.text()
                    logger.error(f"cypher_execution_failed: status={response.status}, error={error_text}")
                    raise Exception(f"Search API error: {response.status} - {error_text}")
    except Exception as e:
        logger.error(f"cypher_request_failed: {str(e)}")
        raise

async def execute_search_documents(
    user_id: str,
    org_id: str,
    filename_pattern: Optional[str] = None,
    file_type: Optional[str] = None,
    category: Optional[str] = None,
    source: Optional[str] = None,
    limit: int = 20
) -> List[Dict[str, Any]]:
    cypher = """
    MATCH (d:Document)
    WHERE d.user_id = $user_id AND d.org_id = $org_id
      AND ($filename_pattern IS NULL OR d.filename CONTAINS $filename_pattern)
      AND ($file_type IS NULL OR d.file_type = $file_type)
      AND ($category IS NULL OR d.category = $category)
      AND ($source IS NULL OR d.source = $source)
    RETURN d
    LIMIT $limit
    """
    
    parameters = {
        "filename_pattern": filename_pattern,
        "file_type": file_type,
        "category": category,
        "source": source,
        "limit": min(limit, 100)
    }
    
    return await _execute_cypher(cypher, parameters, user_id, org_id)

async def execute_get_document_details(
    user_id: str,
    org_id: str,
    internal_object_id: str
) -> List[Dict[str, Any]]:
    cypher = """
    MATCH (d:Document {internal_object_id: $internal_object_id, user_id: $user_id, org_id: $org_id})
    OPTIONAL MATCH (d)-[:HAS_PAGE]->(p:Page)
    OPTIONAL MATCH (p)-[:MENTIONS]->(e:Entity)
    RETURN d, count(DISTINCT p) as page_count, count(DISTINCT e) as entity_count
    """
    
    parameters = {
        "internal_object_id": internal_object_id
    }
    
    return await _execute_cypher(cypher, parameters, user_id, org_id)

async def execute_search_pages_by_content(
    user_id: str,
    org_id: str,
    query_text: str,
    similarity_threshold: float = 0.7,
    limit: int = 10
) -> List[Dict[str, Any]]:
    query_embedding = await _generate_embedding(query_text)
    
    cypher = """
    CALL db.index.vector.queryNodes('page_embedding_index', $limit, $query_embedding)
    YIELD node as p, score
    WHERE p.user_id = $user_id AND p.org_id = $org_id AND score >= $similarity_threshold
    RETURN p, score
    ORDER BY score DESC
    """
    
    parameters = {
        "query_embedding": query_embedding,
        "similarity_threshold": similarity_threshold,
        "limit": min(limit, 50)
    }
    
    return await _execute_cypher(cypher, parameters, user_id, org_id)

async def execute_search_pages_in_document(
    user_id: str,
    org_id: str,
    document_id: str,
    page_number: Optional[int] = None,
    is_tabular: Optional[bool] = None
) -> List[Dict[str, Any]]:
    cypher = """
    MATCH (d:Document {internal_object_id: $document_id, user_id: $user_id, org_id: $org_id})
          -[:HAS_PAGE]->(p:Page)
    WHERE ($page_number IS NULL OR p.page_number = $page_number)
      AND ($is_tabular IS NULL OR p.is_tabular = $is_tabular)
    RETURN p
    ORDER BY p.page_number
    LIMIT 100
    """
    
    parameters = {
        "document_id": document_id,
        "page_number": page_number,
        "is_tabular": is_tabular
    }
    
    return await _execute_cypher(cypher, parameters, user_id, org_id)

async def execute_get_page_details(
    user_id: str,
    org_id: str,
    document_id: str,
    page_number: int
) -> List[Dict[str, Any]]:
    cypher = """
    MATCH (d:Document {internal_object_id: $document_id, user_id: $user_id, org_id: $org_id})
          -[:HAS_PAGE]->(p:Page {page_number: $page_number})
    OPTIONAL MATCH (p)-[:MENTIONS]->(e:Entity)
    RETURN p, collect(e) as entities
    """
    
    parameters = {
        "document_id": document_id,
        "page_number": page_number
    }
    
    return await _execute_cypher(cypher, parameters, user_id, org_id)

async def execute_search_entities_by_semantic(
    user_id: str,
    org_id: str,
    query_text: str,
    entity_type: Optional[str] = None,
    similarity_threshold: float = 0.7,
    limit: int = 10
) -> List[Dict[str, Any]]:
    query_embedding = await _generate_embedding(query_text)
    
    cypher = """
    CALL db.index.vector.queryNodes('entity_embedding_index', $limit, $query_embedding)
    YIELD node as e, score
    WHERE e.user_id = $user_id AND e.org_id = $org_id 
      AND score >= $similarity_threshold
      AND ($entity_type IS NULL OR e.entity_type = $entity_type)
    RETURN e, score
    ORDER BY score DESC
    """
    
    parameters = {
        "query_embedding": query_embedding,
        "entity_type": entity_type,
        "similarity_threshold": similarity_threshold,
        "limit": min(limit, 50)
    }
    
    return await _execute_cypher(cypher, parameters, user_id, org_id)

async def execute_search_entities_by_type(
    user_id: str,
    org_id: str,
    entity_type: str,
    limit: int = 50
) -> List[Dict[str, Any]]:
    cypher = """
    MATCH (e:Entity {entity_type: $entity_type, user_id: $user_id, org_id: $org_id})
    RETURN e
    LIMIT $limit
    """
    
    parameters = {
        "entity_type": entity_type,
        "limit": min(limit, 100)
    }
    
    return await _execute_cypher(cypher, parameters, user_id, org_id)

async def execute_search_entities_by_text(
    user_id: str,
    org_id: str,
    text_pattern: str,
    entity_type: Optional[str] = None,
    limit: int = 30
) -> List[Dict[str, Any]]:
    cypher = """
    MATCH (e:Entity)
    WHERE e.user_id = $user_id AND e.org_id = $org_id
      AND e.text CONTAINS $text_pattern
      AND ($entity_type IS NULL OR e.entity_type = $entity_type)
    RETURN e
    LIMIT $limit
    """
    
    parameters = {
        "text_pattern": text_pattern,
        "entity_type": entity_type,
        "limit": min(limit, 100)
    }
    
    return await _execute_cypher(cypher, parameters, user_id, org_id)

async def execute_get_entity_details(
    user_id: str,
    org_id: str,
    entity_id: str,
    document_id: str
) -> List[Dict[str, Any]]:
    cypher = """
    MATCH (e:Entity {id: $entity_id, document_id: $document_id, user_id: $user_id, org_id: $org_id})
    OPTIONAL MATCH (p:Page)-[:MENTIONS]->(e)
    RETURN e, collect(p) as pages
    """
    
    parameters = {
        "entity_id": entity_id,
        "document_id": document_id
    }
    
    return await _execute_cypher(cypher, parameters, user_id, org_id)

async def execute_find_entity_relationships(
    user_id: str,
    org_id: str,
    entity_id: str,
    document_id: str,
    direction: str = 'both',
    limit: int = 50
) -> List[Dict[str, Any]]:
    if direction == 'outgoing':
        cypher = """
        MATCH (e:Entity {id: $entity_id, document_id: $document_id, user_id: $user_id, org_id: $org_id})
              -[r:RELATIONSHIP]->(target:Entity)
        RETURN e, collect({relationship: r, connected_entity: target}) as relationships
        LIMIT $limit
        """
    elif direction == 'incoming':
        cypher = """
        MATCH (source:Entity)-[r:RELATIONSHIP]->
              (e:Entity {id: $entity_id, document_id: $document_id, user_id: $user_id, org_id: $org_id})
        RETURN e, collect({relationship: r, connected_entity: source}) as relationships
        LIMIT $limit
        """
    else:
        cypher = """
        MATCH (e:Entity {id: $entity_id, document_id: $document_id, user_id: $user_id, org_id: $org_id})
        CALL {
          WITH e
          MATCH (e)-[r:RELATIONSHIP]->(target:Entity)
          RETURN r, target
          UNION
          WITH e
          MATCH (source:Entity)-[r:RELATIONSHIP]->(e)
          RETURN r, source as target
        }
        RETURN e, collect({relationship: r, connected_entity: target}) as relationships
        LIMIT $limit
        """
    
    parameters = {
        "entity_id": entity_id,
        "document_id": document_id,
        "limit": min(limit, 100)
    }
    
    return await _execute_cypher(cypher, parameters, user_id, org_id)

async def execute_search_relationships_by_type(
    user_id: str,
    org_id: str,
    relation_type: str,
    limit: int = 50
) -> List[Dict[str, Any]]:
    cypher = """
    MATCH (source:Entity)-[r:RELATIONSHIP]->(target:Entity)
    WHERE r.user_id = $user_id AND r.org_id = $org_id
      AND r.relation_type = $relation_type
    RETURN source, r, target
    LIMIT $limit
    """
    
    parameters = {
        "relation_type": relation_type,
        "limit": min(limit, 100)
    }
    
    return await _execute_cypher(cypher, parameters, user_id, org_id)

async def execute_search_relationships_semantic(
    user_id: str,
    org_id: str,
    query_text: str,
    similarity_threshold: float = 0.7,
    limit: int = 10
) -> List[Dict[str, Any]]:
    query_embedding = await _generate_embedding(query_text)
    
    cypher = """
    CALL db.index.vector.queryRelationships('relationship_embedding_index', $limit, $query_embedding)
    YIELD relationship as r, score
    WHERE r.user_id = $user_id AND r.org_id = $org_id
      AND score >= $similarity_threshold
    MATCH (source)-[r]->(target)
    RETURN source, r, target, score
    ORDER BY score DESC
    """
    
    parameters = {
        "query_embedding": query_embedding,
        "similarity_threshold": similarity_threshold,
        "limit": min(limit, 50)
    }
    
    return await _execute_cypher(cypher, parameters, user_id, org_id)

async def execute_traverse_entity_graph(
    user_id: str,
    org_id: str,
    entity_id: str,
    document_id: str,
    max_hops: int = 2,
    limit: int = 100
) -> List[Dict[str, Any]]:
    max_hops = max(1, min(max_hops, 3))
    
    cypher = f"""
    MATCH path = (start:Entity {{id: $entity_id, document_id: $document_id, 
                                 user_id: $user_id, org_id: $org_id}})
                  -[:RELATIONSHIP*1..{max_hops}]-(connected:Entity)
    RETURN start, connected, relationships(path) as rels
    LIMIT $limit
    """
    
    parameters = {
        "entity_id": entity_id,
        "document_id": document_id,
        "limit": min(limit, 200)
    }
    
    return await _execute_cypher(cypher, parameters, user_id, org_id)

async def execute_search_columns(
    user_id: str,
    org_id: str,
    column_name_pattern: Optional[str] = None,
    semantic_query: Optional[str] = None,
    document_id: Optional[str] = None,
    limit: int = 20
) -> List[Dict[str, Any]]:
    if semantic_query:
        query_embedding = await _generate_embedding(semantic_query)
        
        cypher = """
        MATCH (c:Column)
        WHERE c.user_id = $user_id AND c.org_id = $org_id
          AND ($column_name_pattern IS NULL OR c.column_name CONTAINS $column_name_pattern)
          AND ($document_id IS NULL OR c.document_id = $document_id)
        WITH collect(c) as keyword_results
        CALL db.index.vector.queryNodes('column_embedding_index', 10, $query_embedding)
        YIELD node as c2, score
        WHERE c2.user_id = $user_id AND c2.org_id = $org_id
          AND ($document_id IS NULL OR c2.document_id = $document_id)
        WITH keyword_results, collect({column: c2, score: score}) as vector_results
        UNWIND keyword_results + [r IN vector_results | r.column] as columns
        RETURN DISTINCT columns as c
        LIMIT $limit
        """
        
        parameters = {
            "column_name_pattern": column_name_pattern,
            "query_embedding": query_embedding,
            "document_id": document_id,
            "limit": min(limit, 50)
        }
    else:
        cypher = """
        MATCH (c:Column)
        WHERE c.user_id = $user_id AND c.org_id = $org_id
          AND ($column_name_pattern IS NULL OR c.column_name CONTAINS $column_name_pattern)
          AND ($document_id IS NULL OR c.document_id = $document_id)
        RETURN c
        LIMIT $limit
        """
        
        parameters = {
            "column_name_pattern": column_name_pattern,
            "document_id": document_id,
            "limit": min(limit, 50)
        }
    
    return await _execute_cypher(cypher, parameters, user_id, org_id)

async def execute_get_column_values(
    user_id: str,
    org_id: str,
    column_name: str,
    document_id: str,
    limit: int = 100
) -> List[Dict[str, Any]]:
    cypher = """
    MATCH (c:Column {column_name: $column_name, document_id: $document_id, 
                      user_id: $user_id, org_id: $org_id})
          -[:HAS_VALUE]->(rv:RowValue)
    RETURN c, rv
    ORDER BY rv.row_index
    LIMIT $limit
    """
    
    parameters = {
        "column_name": column_name,
        "document_id": document_id,
        "limit": min(limit, 200)
    }
    
    return await _execute_cypher(cypher, parameters, user_id, org_id)

async def execute_search_row_values(
    user_id: str,
    org_id: str,
    value_pattern: str,
    column_name: Optional[str] = None,
    document_id: Optional[str] = None,
    limit: int = 50
) -> List[Dict[str, Any]]:
    cypher = """
    MATCH (rv:RowValue)
    WHERE rv.user_id = $user_id AND rv.org_id = $org_id
      AND rv.value CONTAINS $value_pattern
      AND ($column_name IS NULL OR rv.column_name = $column_name)
      AND ($document_id IS NULL OR rv.document_id = $document_id)
    RETURN rv
    LIMIT $limit
    """
    
    parameters = {
        "value_pattern": value_pattern,
        "column_name": column_name,
        "document_id": document_id,
        "limit": min(limit, 200)
    }
    
    return await _execute_cypher(cypher, parameters, user_id, org_id)

async def execute_query_tabular_data(
    user_id: str,
    org_id: str,
    document_id: str,
    sheet_name: Optional[str] = None,
    row_index: Optional[int] = None
) -> List[Dict[str, Any]]:
    cypher = """
    MATCH (d:Document {internal_object_id: $document_id, user_id: $user_id, org_id: $org_id})
          -[:HAS_PAGE]->(p:Page {is_tabular: true})
    WHERE ($sheet_name IS NULL OR p.sheet_name = $sheet_name)
    MATCH (p)-[:MENTIONS]->(c:Column)-[:HAS_VALUE]->(rv:RowValue)
    WHERE ($row_index IS NULL OR rv.row_index = $row_index)
    OPTIONAL MATCH (rv)-[:RELATES_TO]->(related:RowValue)
    RETURN p, c, rv, collect(related) as related_values
    ORDER BY c.column_name, rv.row_index
    LIMIT 500
    """
    
    parameters = {
        "document_id": document_id,
        "sheet_name": sheet_name,
        "row_index": row_index
    }
    
    return await _execute_cypher(cypher, parameters, user_id, org_id)

async def execute_hybrid_search(
    user_id: str,
    org_id: str,
    query_text: str,
    search_nodes: List[str],
    include_relationships: bool = True,
    limit: int = 20
) -> List[Dict[str, Any]]:
    query_embedding = await _generate_embedding(query_text)
    
    results = []
    
    if 'Page' in search_nodes:
        page_cypher = """
        CALL db.index.vector.queryNodes('page_embedding_index', $limit, $query_embedding)
        YIELD node as p, score
        WHERE p.user_id = $user_id AND p.org_id = $org_id
        RETURN 'Page' as node_type, p as node, score
        ORDER BY score DESC
        LIMIT $limit
        """
        page_results = await _execute_cypher(page_cypher, {"query_embedding": query_embedding, "limit": limit}, user_id, org_id)
        results.extend(page_results)
    
    if 'Entity' in search_nodes:
        entity_cypher = """
        CALL db.index.vector.queryNodes('entity_embedding_index', $limit, $query_embedding)
        YIELD node as e, score
        WHERE e.user_id = $user_id AND e.org_id = $org_id
        RETURN 'Entity' as node_type, e as node, score
        ORDER BY score DESC
        LIMIT $limit
        """
        entity_results = await _execute_cypher(entity_cypher, {"query_embedding": query_embedding, "limit": limit}, user_id, org_id)
        results.extend(entity_results)
    
    if 'Column' in search_nodes:
        column_cypher = """
        CALL db.index.vector.queryNodes('column_embedding_index', $limit, $query_embedding)
        YIELD node as c, score
        WHERE c.user_id = $user_id AND c.org_id = $org_id
        RETURN 'Column' as node_type, c as node, score
        ORDER BY score DESC
        LIMIT $limit
        """
        column_results = await _execute_cypher(column_cypher, {"query_embedding": query_embedding, "limit": limit}, user_id, org_id)
        results.extend(column_results)
    
    return _prune_embeddings(results)

async def execute_breadth_first_search(
    user_id: str,
    org_id: str,
    start_node_type: str,
    start_node_id: str,
    max_depth: int = 2,
    relationship_types: Optional[List[str]] = None,
    limit: int = 100
) -> List[Dict[str, Any]]:
    max_depth = max(1, min(max_depth, 3))
    
    if relationship_types:
        rel_filter = f"AND ALL(r IN relationships(path) WHERE type(r) IN {relationship_types})"
    else:
        rel_filter = ""
    
    if start_node_type == 'Document':
        id_field = 'internal_object_id'
    elif start_node_type == 'Page':
        id_field = 'internal_object_id'
    else:
        id_field = 'id'
    
    cypher = f"""
    MATCH path = (start:{start_node_type} {{{id_field}: $start_node_id, user_id: $user_id, org_id: $org_id}})
                 -[*1..{max_depth}]-(connected)
    WHERE connected.user_id = $user_id AND connected.org_id = $org_id
          {rel_filter}
    RETURN start, connected, length(path) as depth, path
    ORDER BY depth
    LIMIT $limit
    """
    
    parameters = {
        "start_node_id": start_node_id,
        "limit": min(limit, 200)
    }
    
    return await _execute_cypher(cypher, parameters, user_id, org_id)

async def execute_get_entity_context(
    user_id: str,
    org_id: str,
    entity_id: str,
    document_id: str,
    include_pages: bool = True,
    include_related_entities: bool = True,
    include_document: bool = True
) -> List[Dict[str, Any]]:
    cypher = """
    MATCH (e:Entity {id: $entity_id, document_id: $document_id, 
                      user_id: $user_id, org_id: $org_id})
    OPTIONAL MATCH (p:Page)-[:MENTIONS]->(e)
    WHERE $include_pages = true
    OPTIONAL MATCH (e)-[r:RELATIONSHIP]-(related:Entity)
    WHERE $include_related_entities = true
    OPTIONAL MATCH (d:Document {internal_object_id: $document_id})-[:HAS_PAGE]->(p)
    WHERE $include_document = true
    RETURN e, 
           collect(DISTINCT p) as pages,
           collect(DISTINCT {entity: related, relationship: r}) as related,
           d as document
    """
    
    parameters = {
        "entity_id": entity_id,
        "document_id": document_id,
        "include_pages": include_pages,
        "include_related_entities": include_related_entities,
        "include_document": include_document
    }
    
    return await _execute_cypher(cypher, parameters, user_id, org_id)
