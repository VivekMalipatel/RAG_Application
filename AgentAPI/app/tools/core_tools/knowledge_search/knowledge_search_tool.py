import yaml
import json
import asyncio
import logging
from pathlib import Path
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from typing import Optional, Dict, Any, List
from langgraph.config import get_stream_writer
from pydantic import Field, BaseModel

logger = logging.getLogger(__name__)

from core.knowledge_search.queries import (
    execute_search_documents,
    execute_get_document_details,
    execute_search_pages_by_content,
    execute_search_pages_in_document,
    execute_get_page_details,
    execute_search_entities_by_semantic,
    execute_search_entities_by_type,
    execute_search_entities_by_text,
    execute_get_entity_details,
    execute_find_entity_relationships,
    execute_search_relationships_by_type,
    execute_search_relationships_semantic,
    execute_traverse_entity_graph,
    execute_search_columns,
    execute_get_column_values,
    execute_search_row_values,
    execute_query_tabular_data,
    execute_hybrid_search,
    execute_breadth_first_search,
    execute_get_entity_context,
    execute_raw_cypher,
)

def get_tool_description(tool_name: str, yaml_filename: str = "description.yaml") -> str:
    yaml_path = Path(__file__).parent / yaml_filename
    with open(yaml_path, 'r') as f:
        descriptions = yaml.safe_load(f)
    return descriptions.get(tool_name, "")

def log_tool_call(tool_name: str, params: Dict[str, Any], user_id: str, org_id: str):
    logger.info(f"[TOOL_CALL] {tool_name} | user_id={user_id} | org_id={org_id} | params={json.dumps(params, default=str)}")

def log_tool_success(tool_name: str, result_count: int, user_id: str, org_id: str):
    logger.info(f"[TOOL_SUCCESS] {tool_name} | user_id={user_id} | org_id={org_id} | results={result_count}")

def log_tool_error(tool_name: str, error: Exception, user_id: str, org_id: str):
    logger.error(f"[TOOL_ERROR] {tool_name} | user_id={user_id} | org_id={org_id} | error={str(error)}", exc_info=True)

def _ensure_image_payload(image_value: Any) -> Optional[Dict[str, Any]]:
    if image_value is None:
        return None
    if isinstance(image_value, dict):
        url = image_value.get("url")
        if isinstance(url, str) and url.strip():
            return {"url": url.strip()}
        return None
    if isinstance(image_value, str) and image_value.strip():
        return {"url": image_value.strip()}
    return None


def _safe_json_loads(value: str) -> Optional[Any]:
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return None


def _convert_content_block(block: Any) -> List[Dict[str, Any]]:
    if block is None:
        return []

    if isinstance(block, dict):
        block_type = block.get("type")
        if block_type in {"text", "input_text"}:
            text_value = block.get("text") or block.get("value") or block.get("content")
            if isinstance(text_value, str) and text_value.strip():
                return [{"type": "text", "text": text_value.strip()}]
        if block_type in {"image_url", "input_image"}:
            image_value = block.get("image_url") or block.get("image") or block.get("content")
            payload = _ensure_image_payload(image_value)
            if payload:
                return [{"type": "image_url", "image_url": payload}]
        if block_type == "file":
            name = block.get("name") or block.get("filename") or "file"
            return [{"type": "text", "text": f"[file attachment: {name}]"}]
        if block_type in {"audio_url", "input_audio", "video_url", "input_video"}:
            descriptor = block_type.replace("_", " ")
            return [{"type": "text", "text": f"[{descriptor} content omitted]"}]

    if isinstance(block, str) and block.strip():
        return [{"type": "text", "text": block.strip()}]

    return []


def _collect_blocks_from_message(message: Any) -> List[Dict[str, Any]]:
    if not isinstance(message, dict):
        return []

    content = message.get("content")
    if isinstance(content, list):
        blocks: List[Dict[str, Any]] = []
        for block in content:
            blocks.extend(_convert_content_block(block))
        return blocks

    if isinstance(content, str) and content.strip():
        return [{"type": "text", "text": content.strip()}]

    return []


def _format_messages_payload(payload: Any) -> List[Dict[str, Any]]:
    if payload is None:
        return []

    parsed_payload = payload
    if isinstance(payload, str):
        parsed = _safe_json_loads(payload)
        if parsed is None:
            text = payload.strip()
            return ([{"type": "text", "text": text}] if text else [])
        parsed_payload = parsed

    if isinstance(parsed_payload, list):
        if parsed_payload and isinstance(parsed_payload[0], dict) and "role" in parsed_payload[0]:
            blocks: List[Dict[str, Any]] = []
            for message in parsed_payload:
                blocks.extend(_collect_blocks_from_message(message))
            return blocks

        blocks: List[Dict[str, Any]] = []
        for item in parsed_payload:
            blocks.extend(_convert_content_block(item))
        return blocks

    if isinstance(parsed_payload, dict):
        return _collect_blocks_from_message(parsed_payload)

    return []


def _append_metadata_block(blocks: List[Dict[str, Any]], label: str, metadata: Dict[str, Any]) -> None:
    parts = []
    for key, value in metadata.items():
        if value in (None, "", [], {}):
            continue
        parts.append(f"{key}: {value}")
    if parts:
        blocks.append({"type": "text", "text": f"[{label}] {' | '.join(parts)}"})


def _format_document_node(document: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(document, dict):
        return []

    blocks: List[Dict[str, Any]] = []
    for key in ("summary", "description", "notes"):
        value = document.get(key)
        if isinstance(value, str) and value.strip():
            blocks.append({"type": "text", "text": value.strip()})

    metadata = {
        "filename": document.get("filename"),
        "source": document.get("source"),
        "category": document.get("category"),
        "file_type": document.get("file_type"),
        "internal_object_id": document.get("internal_object_id"),
        "task_id": document.get("task_id"),
    }
    _append_metadata_block(blocks, "DOCUMENT METADATA", metadata)
    return blocks


def _format_page_node(page: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(page, dict):
        return []

    blocks: List[Dict[str, Any]] = []
    blocks.extend(_format_messages_payload(page.get("content")))

    summary = page.get("summary")
    if not blocks and isinstance(summary, str) and summary.strip():
        blocks.append({"type": "text", "text": summary.strip()})

    metadata = {
        "page_number": page.get("page_number"),
        "sheet_name": page.get("sheet_name"),
        "image_s3_url": page.get("image_s3_url"),
        "total_rows": page.get("total_rows"),
        "total_columns": page.get("total_columns"),
        "is_tabular": page.get("is_tabular"),
    }
    _append_metadata_block(blocks, "PAGE METADATA", metadata)
    return blocks


def _format_entity_node(entity: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(entity, dict):
        return []

    blocks: List[Dict[str, Any]] = []
    profile = entity.get("entity_profile")
    if isinstance(profile, str) and profile.strip():
        blocks.append({"type": "text", "text": profile.strip()})

    text_value = entity.get("text")
    if isinstance(text_value, str) and text_value.strip():
        blocks.append({"type": "text", "text": text_value.strip()})

    metadata = {
        "entity_id": entity.get("id"),
        "entity_type": entity.get("entity_type"),
        "document_id": entity.get("document_id"),
    }
    _append_metadata_block(blocks, "ENTITY METADATA", metadata)
    return blocks


def _format_relationship_edge(relationship: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(relationship, dict):
        return []

    blocks: List[Dict[str, Any]] = []
    profile = relationship.get("relation_profile")
    if isinstance(profile, str) and profile.strip():
        blocks.append({"type": "text", "text": profile.strip()})

    metadata = {
        "relation_type": relationship.get("relation_type"),
    }
    _append_metadata_block(blocks, "RELATIONSHIP METADATA", metadata)
    return blocks


def _format_column_node(column: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(column, dict):
        return []

    blocks: List[Dict[str, Any]] = []
    profile = column.get("column_profile")
    if isinstance(profile, str) and profile.strip():
        blocks.append({"type": "text", "text": profile.strip()})

    metadata = {
        "column_name": column.get("column_name"),
    }
    _append_metadata_block(blocks, "COLUMN METADATA", metadata)
    return blocks


def _format_row_value_node(row_value: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(row_value, dict):
        return []

    blocks: List[Dict[str, Any]] = []
    value = row_value.get("value")
    if value not in (None, ""):
        blocks.append({"type": "text", "text": str(value)})

    metadata = {
        "row_index": row_value.get("row_index"),
        "column_name": row_value.get("column_name"),
    }
    _append_metadata_block(blocks, "ROW VALUE METADATA", metadata)
    return blocks


def _format_generic_metadata(key: str, value: Any) -> List[str]:
    if value in (None, "", [], {}):
        return []
    if isinstance(value, (dict, list)):
        return []
    return [f"{key}: {value}"]


async def process_single_record(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    metadata_entries: List[str] = []

    for key, value in record.items():
        if key in {"content", "messages"}:
            blocks.extend(_format_messages_payload(value))
            continue

        if key in {"document", "d"}:
            blocks.extend(_format_document_node(value))
            continue

        if key in {"page", "p"}:
            blocks.extend(_format_page_node(value))
            continue

        if key == "pages" and isinstance(value, list):
            for page in value:
                blocks.extend(_format_page_node(page))
            continue

        if key in {"entity", "e"}:
            blocks.extend(_format_entity_node(value))
            continue

        if key in {"entities", "related_entities"} and isinstance(value, list):
            for entity in value:
                blocks.extend(_format_entity_node(entity))
            continue

        if key in {"source", "target", "start", "connected"}:
            blocks.extend(_format_entity_node(value))
            continue

        if key == "related" and isinstance(value, list):
            for entry in value:
                if not isinstance(entry, dict):
                    continue
                related_entity = entry.get("entity") or entry.get("related")
                relationship = entry.get("relationship")
                if related_entity:
                    blocks.extend(_format_entity_node(related_entity))
                if relationship:
                    blocks.extend(_format_relationship_edge(relationship))
            continue

        if key in {"relationship", "relationships", "rels", "r"}:
            relationships = value if isinstance(value, list) else [value]
            for relationship_entry in relationships:
                if isinstance(relationship_entry, dict) and (
                    "relationship" in relationship_entry or "connected_entity" in relationship_entry
                ):
                    relation = relationship_entry.get("relationship") or relationship_entry.get("r")
                    if relation:
                        blocks.extend(_format_relationship_edge(relation))
                    connected_entity = (
                        relationship_entry.get("connected_entity")
                        or relationship_entry.get("entity")
                        or relationship_entry.get("target")
                        or relationship_entry.get("source")
                    )
                    if connected_entity:
                        blocks.extend(_format_entity_node(connected_entity))
                    continue

                blocks.extend(_format_relationship_edge(relationship_entry))
            continue

        if key in {"column", "columns"}:
            columns = value if isinstance(value, list) else [value]
            for column in columns:
                blocks.extend(_format_column_node(column))
            continue

        if key in {"row", "rows", "row_values"}:
            rows = value if isinstance(value, list) else [value]
            for row_value in rows:
                blocks.extend(_format_row_value_node(row_value))
            continue

        metadata_entries.extend(_format_generic_metadata(key, value))

    if metadata_entries:
        blocks.append({"type": "text", "text": f"[RECORD METADATA] {' | '.join(metadata_entries)}"})

    if not blocks:
        blocks.append({"type": "text", "text": json.dumps(record, indent=2, default=str)})

    return blocks

async def format_neo4j_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not results:
        return []
    
    processed_records = await asyncio.gather(*[process_single_record(record) for record in results])
    
    formatted_results = []
    for record_content in processed_records:
        formatted_results.extend(record_content)
    
    return formatted_results

class SearchDocumentsSchema(BaseModel):
    filename_pattern: Optional[str] = Field(default=None, description="Partial match on filename (uses CONTAINS)")
    file_type: Optional[str] = Field(default=None, description="Exact file type filter (pdf, csv, jpeg, etc.)")
    category: Optional[str] = Field(default=None, description="Category filter (structured/unstructured)")
    source: Optional[str] = Field(default=None, description="Source system filter")
    limit: int = Field(default=20, description="Max results (default 20, max 100)")

@tool(
    name_or_callable="search_documents",
    description=get_tool_description("search_documents"),
    args_schema=SearchDocumentsSchema,
    response_format="content"
)
async def search_documents(
    filename_pattern: Optional[str] = None,
    file_type: Optional[str] = None,
    category: Optional[str] = None,
    source: Optional[str] = None,
    limit: int = 20,
    config: RunnableConfig = None
) -> List[Dict[str, Any]]:
    user_id = config.get("configurable", {}).get("user_id")
    org_id = config.get("configurable", {}).get("org_id")
    
    if not user_id or not org_id:
        error_msg = "Error: user_id and org_id are required in config"
        logger.error(f"[TOOL_ERROR] search_documents | Missing credentials")
        return [{"type": "text", "text": error_msg}]
    
    log_tool_call("search_documents", {
        "filename_pattern": filename_pattern,
        "file_type": file_type,
        "category": category,
        "source": source,
        "limit": limit
    }, user_id, org_id)
    
    try:
        writer = get_stream_writer()
        writer(f"search_documents: filename_pattern={filename_pattern}, file_type={file_type}")
        
        results = await execute_search_documents(
            user_id=user_id,
            org_id=org_id,
            filename_pattern=filename_pattern,
            file_type=file_type,
            category=category,
            source=source,
            limit=limit
        )
        
        formatted_results = await format_neo4j_results(results)
        log_tool_success("search_documents", len(results), user_id, org_id)
        return formatted_results
    except Exception as e:
        log_tool_error("search_documents", e, user_id, org_id)
        return [{"type": "text", "text": f"Error executing search_documents: {str(e)}"}]

class GetDocumentDetailsSchema(BaseModel):
    internal_object_id: str = Field(description="Document internal object ID")

@tool(
    name_or_callable="get_document_details",
    description=get_tool_description("get_document_details"),
    args_schema=GetDocumentDetailsSchema,
    response_format="content"
)
async def get_document_details(
    internal_object_id: str,
    config: RunnableConfig = None
) -> List[Dict[str, Any]]:
    user_id = config.get("configurable", {}).get("user_id")
    org_id = config.get("configurable", {}).get("org_id")
    
    if not user_id or not org_id:
        error_msg = "Error: user_id and org_id are required in config"
        logger.error(f"[TOOL_ERROR] get_document_details | Missing credentials")
        return [{"type": "text", "text": error_msg}]
    
    log_tool_call("get_document_details", {"internal_object_id": internal_object_id}, user_id, org_id)
    
    try:
        writer = get_stream_writer()
        writer(f"get_document_details: {internal_object_id}")
        
        results = await execute_get_document_details(
            user_id=user_id,
            org_id=org_id,
            internal_object_id=internal_object_id
        )
        
        formatted_results = await format_neo4j_results(results)
        log_tool_success("get_document_details", len(results), user_id, org_id)
        return formatted_results
    except Exception as e:
        log_tool_error("get_document_details", e, user_id, org_id)
        return [{"type": "text", "text": f"Error executing get_document_details: {str(e)}"}]

class SearchPagesByContentSchema(BaseModel):
    query_text: str = Field(description="Natural language search query")
    similarity_threshold: float = Field(default=0.7, description="Minimum similarity (0-1)")
    limit: int = Field(default=10, description="Max results")

@tool(
    name_or_callable="search_pages_by_content",
    description=get_tool_description("search_pages_by_content"),
    args_schema=SearchPagesByContentSchema,
    response_format="content"
)
async def search_pages_by_content(
    query_text: str,
    similarity_threshold: float = 0.7,
    limit: int = 10,
    config: RunnableConfig = None
) -> List[Dict[str, Any]]:
    user_id = config.get("configurable", {}).get("user_id")
    org_id = config.get("configurable", {}).get("org_id")
    
    if not user_id or not org_id:
        error_msg = "Error: user_id and org_id are required in config"
        logger.error(f"[TOOL_ERROR] search_pages_by_content | Missing credentials")
        return [{"type": "text", "text": error_msg}]
    
    log_tool_call("search_pages_by_content", {"query_text": query_text[:100], "similarity_threshold": similarity_threshold, "limit": limit}, user_id, org_id)
    
    try:
        writer = get_stream_writer()
        writer(f"search_pages_by_content: {query_text[:50]}")
        
        results = await execute_search_pages_by_content(
            user_id=user_id,
            org_id=org_id,
            query_text=query_text,
            similarity_threshold=similarity_threshold,
            limit=limit
        )
        
        formatted_results = await format_neo4j_results(results)
        log_tool_success("search_pages_by_content", len(results), user_id, org_id)
        return formatted_results
    except Exception as e:
        log_tool_error("search_pages_by_content", e, user_id, org_id)
        return [{"type": "text", "text": f"Error executing search_pages_by_content: {str(e)}"}]

class SearchPagesInDocumentSchema(BaseModel):
    document_id: str = Field(description="Document internal object ID")
    page_number: Optional[int] = Field(default=None, description="Specific page number filter")
    is_tabular: Optional[bool] = Field(default=None, description="Filter tabular vs non-tabular pages")

@tool(
    name_or_callable="search_pages_in_document",
    description=get_tool_description("search_pages_in_document"),
    args_schema=SearchPagesInDocumentSchema,
    response_format="content"
)
async def search_pages_in_document(
    document_id: str,
    page_number: Optional[int] = None,
    is_tabular: Optional[bool] = None,
    config: RunnableConfig = None
) -> List[Dict[str, Any]]:
    user_id = config.get("configurable", {}).get("user_id")
    org_id = config.get("configurable", {}).get("org_id")
    
    if not user_id or not org_id:
        error_msg = "Error: user_id and org_id are required in config"
        logger.error(f"[TOOL_ERROR] search_pages_in_document | Missing credentials")
        return [{"type": "text", "text": error_msg}]
    
    log_tool_call("search_pages_in_document", {"document_id": document_id, "page_number": page_number, "is_tabular": is_tabular}, user_id, org_id)
    
    try:
        writer = get_stream_writer()
        writer(f"search_pages_in_document: {document_id}")
        
        results = await execute_search_pages_in_document(
            user_id=user_id,
            org_id=org_id,
            document_id=document_id,
            page_number=page_number,
            is_tabular=is_tabular
        )
        
        formatted_results = await format_neo4j_results(results)
        log_tool_success("search_pages_in_document", len(results), user_id, org_id)
        return formatted_results
    except Exception as e:
        log_tool_error("search_pages_in_document", e, user_id, org_id)
        return [{"type": "text", "text": f"Error executing search_pages_in_document: {str(e)}"}]

class GetPageDetailsSchema(BaseModel):
    document_id: str = Field(description="Document internal object ID")
    page_number: int = Field(description="Page number")

@tool(
    name_or_callable="get_page_details",
    description=get_tool_description("get_page_details"),
    args_schema=GetPageDetailsSchema,
    response_format="content"
)
async def get_page_details(
    document_id: str,
    page_number: int,
    config: RunnableConfig = None
) -> List[Dict[str, Any]]:
    user_id = config.get("configurable", {}).get("user_id")
    org_id = config.get("configurable", {}).get("org_id")
    
    if not user_id or not org_id:
        error_msg = "Error: user_id and org_id are required in config"
        logger.error(f"[TOOL_ERROR] get_page_details | Missing credentials")
        return [{"type": "text", "text": error_msg}]
    
    log_tool_call("get_page_details", {"document_id": document_id, "page_number": page_number}, user_id, org_id)
    
    try:
        writer = get_stream_writer()
        writer(f"get_page_details: {document_id}, page {page_number}")
        
        results = await execute_get_page_details(
            user_id=user_id,
            org_id=org_id,
            document_id=document_id,
            page_number=page_number
        )
        
        
        formatted_results = await format_neo4j_results(results)
        log_tool_success("get_page_details", len(results), user_id, org_id)
        return formatted_results
    except Exception as e:
        log_tool_error("get_page_details", e, user_id, org_id)
        return [{"type": "text", "text": f"Error executing get_page_details: {str(e)}"}]

class SearchEntitiesBySemanticSchema(BaseModel):
    query_text: str = Field(description="Natural language entity search query")
    entity_type: Optional[str] = Field(default=None, description="Filter by entity type")
    similarity_threshold: float = Field(default=0.7, description="Minimum similarity")
    limit: int = Field(default=10, description="Max results")

@tool(
    name_or_callable="search_entities_by_semantic",
    description=get_tool_description("search_entities_by_semantic"),
    args_schema=SearchEntitiesBySemanticSchema,
    response_format="content"
)
async def search_entities_by_semantic(
    query_text: str,
    entity_type: Optional[str] = None,
    similarity_threshold: float = 0.7,
    limit: int = 10,
    config: RunnableConfig = None
) -> List[Dict[str, Any]]:
    user_id = config.get("configurable", {}).get("user_id")
    org_id = config.get("configurable", {}).get("org_id")
    
    if not user_id or not org_id:
        error_msg = "Error: user_id and org_id are required in config"
        logger.error(f"[TOOL_ERROR] search_entities_by_semantic | Missing credentials")
        return [{"type": "text", "text": error_msg}]
    
    log_tool_call("search_entities_by_semantic", {"query_text": query_text[:100], "entity_type": entity_type, "similarity_threshold": similarity_threshold, "limit": limit}, user_id, org_id)
    
    try:
        writer = get_stream_writer()
        writer(f"search_entities_by_semantic: {query_text[:50]}")
        
        results = await execute_search_entities_by_semantic(
            user_id=user_id,
            org_id=org_id,
            query_text=query_text,
            entity_type=entity_type,
            similarity_threshold=similarity_threshold,
            limit=limit
        )
        
        
        formatted_results = await format_neo4j_results(results)
        log_tool_success("search_entities_by_semantic", len(results), user_id, org_id)
        return formatted_results
    except Exception as e:
        log_tool_error("search_entities_by_semantic", e, user_id, org_id)
        return [{"type": "text", "text": f"Error executing search_entities_by_semantic: {str(e)}"}]

class SearchEntitiesByTypeSchema(BaseModel):
    entity_type: str = Field(description="Entity type to search for")
    limit: int = Field(default=50, description="Max results")

@tool(
    name_or_callable="search_entities_by_type",
    description=get_tool_description("search_entities_by_type"),
    args_schema=SearchEntitiesByTypeSchema,
    response_format="content"
)
async def search_entities_by_type(
    entity_type: str,
    limit: int = 50,
    config: RunnableConfig = None
) -> List[Dict[str, Any]]:
    user_id = config.get("configurable", {}).get("user_id")
    org_id = config.get("configurable", {}).get("org_id")
    
    if not user_id or not org_id:
        error_msg = "Error: user_id and org_id are required in config"
        logger.error(f"[TOOL_ERROR] search_entities_by_type | Missing credentials")
        return [{"type": "text", "text": error_msg}]
    
    log_tool_call("search_entities_by_type", {"entity_type": entity_type, "limit": limit}, user_id, org_id)
    
    try:
        writer = get_stream_writer()
        writer(f"search_entities_by_type: {entity_type}")
        
        results = await execute_search_entities_by_type(
            user_id=user_id,
            org_id=org_id,
            entity_type=entity_type,
            limit=limit
        )
        
        
        formatted_results = await format_neo4j_results(results)
        log_tool_success("search_entities_by_type", len(results), user_id, org_id)
        return formatted_results
    except Exception as e:
        log_tool_error("search_entities_by_type", e, user_id, org_id)
        return [{"type": "text", "text": f"Error executing search_entities_by_type: {str(e)}"}]

class SearchEntitiesByTextSchema(BaseModel):
    text_pattern: str = Field(description="Text pattern to match (uses CONTAINS)")
    entity_type: Optional[str] = Field(default=None, description="Filter by entity type")
    limit: int = Field(default=30, description="Max results")

@tool(
    name_or_callable="search_entities_by_text",
    description=get_tool_description("search_entities_by_text"),
    args_schema=SearchEntitiesByTextSchema,
    response_format="content"
)
async def search_entities_by_text(
    text_pattern: str,
    entity_type: Optional[str] = None,
    limit: int = 30,
    config: RunnableConfig = None
) -> List[Dict[str, Any]]:
    user_id = config.get("configurable", {}).get("user_id")
    org_id = config.get("configurable", {}).get("org_id")
    
    if not user_id or not org_id:
        error_msg = "Error: user_id and org_id are required in config"
        logger.error(f"[TOOL_ERROR] search_entities_by_text | Missing credentials")
        return [{"type": "text", "text": error_msg}]
    
    try:
        log_tool_call("search_entities_by_text", {"text_pattern": text_pattern, "entity_type": entity_type, "limit": limit}, user_id, org_id)

        writer = get_stream_writer()
        writer(f"search_entities_by_text: {text_pattern}")
        
        results = await execute_search_entities_by_text(
            user_id=user_id,
            org_id=org_id,
            text_pattern=text_pattern,
            entity_type=entity_type,
            limit=limit
        )
        
        
        formatted_results = await format_neo4j_results(results)
        log_tool_success("search_entities_by_text", len(results), user_id, org_id)
        return formatted_results
    except Exception as e:
        log_tool_error("search_entities_by_text", e, user_id, org_id)
        return [{"type": "text", "text": f"Error executing search_entities_by_text: {str(e)}"}]

class GetEntityDetailsSchema(BaseModel):
    entity_id: str = Field(description="Entity ID")
    document_id: str = Field(description="Document internal object ID")

@tool(
    name_or_callable="get_entity_details",
    description=get_tool_description("get_entity_details"),
    args_schema=GetEntityDetailsSchema,
    response_format="content"
)
async def get_entity_details(
    entity_id: str,
    document_id: str,
    config: RunnableConfig = None
) -> List[Dict[str, Any]]:
    user_id = config.get("configurable", {}).get("user_id")
    org_id = config.get("configurable", {}).get("org_id")
    
    if not user_id or not org_id:
        error_msg = "Error: user_id and org_id are required in config"
        logger.error(f"[TOOL_ERROR] get_entity_details | Missing credentials")
        return [{"type": "text", "text": error_msg}]
    
    log_tool_call("get_entity_details", {"entity_id": entity_id}, user_id, org_id)
    
    try:
        writer = get_stream_writer()
        writer(f"get_entity_details: {entity_id}")
        
        results = await execute_get_entity_details(
            user_id=user_id,
            org_id=org_id,
            entity_id=entity_id,
            document_id=document_id
        )
        
        
        formatted_results = await format_neo4j_results(results)
        log_tool_success("get_entity_details", len(results), user_id, org_id)
        return formatted_results
    except Exception as e:
        log_tool_error("get_entity_details", e, user_id, org_id)
        return [{"type": "text", "text": f"Error executing get_entity_details: {str(e)}"}]

class FindEntityRelationshipsSchema(BaseModel):
    entity_id: str = Field(description="Entity ID")
    document_id: str = Field(description="Document internal object ID")
    direction: str = Field(default='both', description="'outgoing', 'incoming', or 'both'")
    limit: int = Field(default=50, description="Max results")

@tool(
    name_or_callable="find_entity_relationships",
    description=get_tool_description("find_entity_relationships"),
    args_schema=FindEntityRelationshipsSchema,
    response_format="content"
)
async def find_entity_relationships(
    entity_id: str,
    document_id: str,
    direction: str = 'both',
    limit: int = 50,
    config: RunnableConfig = None
) -> List[Dict[str, Any]]:
    user_id = config.get("configurable", {}).get("user_id")
    org_id = config.get("configurable", {}).get("org_id")
    
    if not user_id or not org_id:
        error_msg = "Error: user_id and org_id are required in config"
        logger.error(f"[TOOL_ERROR] find_entity_relationships | Missing credentials")
        return [{"type": "text", "text": error_msg}]
    
    try:
        log_tool_call(
            "find_entity_relationships",
            {
                "entity_id": entity_id,
                "document_id": document_id,
                "direction": direction,
                "limit": limit,
            },
            user_id,
            org_id,
        )

        writer = get_stream_writer()
        writer(f"find_entity_relationships: {entity_id}, direction={direction}")
        
        results = await execute_find_entity_relationships(
            user_id=user_id,
            org_id=org_id,
            entity_id=entity_id,
            document_id=document_id,
            direction=direction,
            limit=limit
        )
        
        
        formatted_results = await format_neo4j_results(results)
        log_tool_success("find_entity_relationships", len(results), user_id, org_id)
        return formatted_results
    except Exception as e:
        log_tool_error("find_entity_relationships", e, user_id, org_id)
        return [{"type": "text", "text": f"Error executing find_entity_relationships: {str(e)}"}]

class SearchRelationshipsByTypeSchema(BaseModel):
    relation_type: str = Field(description="Relationship type to search")
    limit: int = Field(default=50, description="Max results")

@tool(
    name_or_callable="search_relationships_by_type",
    description=get_tool_description("search_relationships_by_type"),
    args_schema=SearchRelationshipsByTypeSchema,
    response_format="content"
)
async def search_relationships_by_type(
    relation_type: str,
    limit: int = 50,
    config: RunnableConfig = None
) -> List[Dict[str, Any]]:
    user_id = config.get("configurable", {}).get("user_id")
    org_id = config.get("configurable", {}).get("org_id")
    
    if not user_id or not org_id:
        error_msg = "Error: user_id and org_id are required in config"
        logger.error(f"[TOOL_ERROR] search_relationships_by_type | Missing credentials")
        return [{"type": "text", "text": error_msg}]
    
    try:
        log_tool_call(
            "search_relationships_by_type",
            {"relation_type": relation_type, "limit": limit},
            user_id,
            org_id,
        )

        writer = get_stream_writer()
        writer(f"search_relationships_by_type: {relation_type}")
        
        results = await execute_search_relationships_by_type(
            user_id=user_id,
            org_id=org_id,
            relation_type=relation_type,
            limit=limit
        )
        
        
        formatted_results = await format_neo4j_results(results)
        log_tool_success("search_relationships_by_type", len(results), user_id, org_id)
        return formatted_results
    except Exception as e:
        log_tool_error("search_relationships_by_type", e, user_id, org_id)
        return [{"type": "text", "text": f"Error executing search_relationships_by_type: {str(e)}"}]

class SearchRelationshipsSemanticSchema(BaseModel):
    query_text: str = Field(description="Relationship description query")
    similarity_threshold: float = Field(default=0.7, description="Minimum similarity")
    limit: int = Field(default=10, description="Max results")

@tool(
    name_or_callable="search_relationships_semantic",
    description=get_tool_description("search_relationships_semantic"),
    args_schema=SearchRelationshipsSemanticSchema,
    response_format="content"
)
async def search_relationships_semantic(
    query_text: str,
    similarity_threshold: float = 0.7,
    limit: int = 10,
    config: RunnableConfig = None
) -> List[Dict[str, Any]]:
    user_id = config.get("configurable", {}).get("user_id")
    org_id = config.get("configurable", {}).get("org_id")
    
    if not user_id or not org_id:
        error_msg = "Error: user_id and org_id are required in config"
        logger.error(f"[TOOL_ERROR] search_relationships_semantic | Missing credentials")
        return [{"type": "text", "text": error_msg}]
    
    log_tool_call("search_relationships_semantic", {"query_text": query_text[:100], "limit": limit}, user_id, org_id)
    
    try:
        writer = get_stream_writer()
        writer(f"search_relationships_semantic: {query_text[:50]}")
        
        results = await execute_search_relationships_semantic(
            user_id=user_id,
            org_id=org_id,
            query_text=query_text,
            similarity_threshold=similarity_threshold,
            limit=limit
        )
        
        
        formatted_results = await format_neo4j_results(results)
        log_tool_success("search_relationships_semantic", len(results), user_id, org_id)
        return formatted_results
    except Exception as e:
        log_tool_error("search_relationships_semantic", e, user_id, org_id)
        return [{"type": "text", "text": f"Error executing search_relationships_semantic: {str(e)}"}]

class TraverseEntityGraphSchema(BaseModel):
    entity_id: str = Field(description="Starting entity ID")
    document_id: str = Field(description="Document internal object ID")
    max_hops: int = Field(default=2, description="Max hops (1-3)")
    limit: int = Field(default=100, description="Max results")

@tool(
    name_or_callable="traverse_entity_graph",
    description=get_tool_description("traverse_entity_graph"),
    args_schema=TraverseEntityGraphSchema,
    response_format="content"
)
async def traverse_entity_graph(
    entity_id: str,
    document_id: str,
    max_hops: int = 2,
    limit: int = 100,
    config: RunnableConfig = None
) -> List[Dict[str, Any]]:
    user_id = config.get("configurable", {}).get("user_id")
    org_id = config.get("configurable", {}).get("org_id")
    
    if not user_id or not org_id:
        error_msg = "Error: user_id and org_id are required in config"
        logger.error(f"[TOOL_ERROR] traverse_entity_graph | Missing credentials")
        return [{"type": "text", "text": error_msg}]
    
    try:
        log_tool_call(
            "traverse_entity_graph",
            {
                "entity_id": entity_id,
                "document_id": document_id,
                "max_hops": max_hops,
                "limit": limit,
            },
            user_id,
            org_id,
        )

        writer = get_stream_writer()
        writer(f"traverse_entity_graph: {entity_id}, max_hops={max_hops}")
        
        results = await execute_traverse_entity_graph(
            user_id=user_id,
            org_id=org_id,
            entity_id=entity_id,
            document_id=document_id,
            max_hops=max_hops,
            limit=limit
        )
        
        
        formatted_results = await format_neo4j_results(results)
        log_tool_success("traverse_entity_graph", len(results), user_id, org_id)
        return formatted_results
    except Exception as e:
        log_tool_error("traverse_entity_graph", e, user_id, org_id)
        return [{"type": "text", "text": f"Error executing traverse_entity_graph: {str(e)}"}]

class SearchColumnsSchema(BaseModel):
    column_name_pattern: Optional[str] = Field(default=None, description="Partial column name match")
    semantic_query: Optional[str] = Field(default=None, description="Semantic search on column profile")
    document_id: Optional[str] = Field(default=None, description="Limit to specific document")
    limit: int = Field(default=20, description="Max results")

@tool(
    name_or_callable="search_columns",
    description=get_tool_description("search_columns"),
    args_schema=SearchColumnsSchema,
    response_format="content"
)
async def search_columns(
    column_name_pattern: Optional[str] = None,
    semantic_query: Optional[str] = None,
    document_id: Optional[str] = None,
    limit: int = 20,
    config: RunnableConfig = None
) -> List[Dict[str, Any]]:
    user_id = config.get("configurable", {}).get("user_id")
    org_id = config.get("configurable", {}).get("org_id")
    
    if not user_id or not org_id:
        error_msg = "Error: user_id and org_id are required in config"
        logger.error(f"[TOOL_ERROR] search_columns | Missing credentials")
        return [{"type": "text", "text": error_msg}]
    
    try:
        log_tool_call(
            "search_columns",
            {
                "column_name_pattern": column_name_pattern,
                "semantic_query": semantic_query,
                "document_id": document_id,
                "limit": limit,
            },
            user_id,
            org_id,
        )
        writer = get_stream_writer()
        writer(f"search_columns: pattern={column_name_pattern}, semantic={semantic_query is not None}")
        
        results = await execute_search_columns(
            user_id=user_id,
            org_id=org_id,
            column_name_pattern=column_name_pattern,
            semantic_query=semantic_query,
            document_id=document_id,
            limit=limit
        )
        
        
        formatted_results = await format_neo4j_results(results)
        log_tool_success("search_columns", len(results), user_id, org_id)
        return formatted_results
    except Exception as e:
        log_tool_error("search_columns", e, user_id, org_id)
        return [{"type": "text", "text": f"Error executing search_columns: {str(e)}"}]

class GetColumnValuesSchema(BaseModel):
    column_name: str = Field(description="Column name")
    document_id: str = Field(description="Document internal object ID")
    limit: int = Field(default=100, description="Max results")

@tool(
    name_or_callable="get_column_values",
    description=get_tool_description("get_column_values"),
    args_schema=GetColumnValuesSchema,
    response_format="content"
)
async def get_column_values(
    column_name: str,
    document_id: str,
    limit: int = 100,
    config: RunnableConfig = None
) -> List[Dict[str, Any]]:
    user_id = config.get("configurable", {}).get("user_id")
    org_id = config.get("configurable", {}).get("org_id")
    
    if not user_id or not org_id:
        error_msg = "Error: user_id and org_id are required in config"
        logger.error(f"[TOOL_ERROR] get_column_values | Missing credentials")
        return [{"type": "text", "text": error_msg}]
    
    try:
        log_tool_call(
            "get_column_values",
            {
                "column_name": column_name,
                "document_id": document_id,
                "limit": limit,
            },
            user_id,
            org_id,
        )

        writer = get_stream_writer()
        writer(f"get_column_values: {column_name}")
        
        results = await execute_get_column_values(
            user_id=user_id,
            org_id=org_id,
            column_name=column_name,
            document_id=document_id,
            limit=limit
        )
        
        
        formatted_results = await format_neo4j_results(results)
        log_tool_success("get_column_values", len(results), user_id, org_id)
        return formatted_results
    except Exception as e:
        log_tool_error("get_column_values", e, user_id, org_id)
        return [{"type": "text", "text": f"Error executing get_column_values: {str(e)}"}]

class SearchRowValuesSchema(BaseModel):
    value_pattern: str = Field(description="Value pattern to match (uses CONTAINS)")
    column_name: Optional[str] = Field(default=None, description="Limit to specific column")
    document_id: Optional[str] = Field(default=None, description="Limit to specific document")
    limit: int = Field(default=50, description="Max results")

@tool(
    name_or_callable="search_row_values",
    description=get_tool_description("search_row_values"),
    args_schema=SearchRowValuesSchema,
    response_format="content"
)
async def search_row_values(
    value_pattern: str,
    column_name: Optional[str] = None,
    document_id: Optional[str] = None,
    limit: int = 50,
    config: RunnableConfig = None
) -> List[Dict[str, Any]]:
    user_id = config.get("configurable", {}).get("user_id")
    org_id = config.get("configurable", {}).get("org_id")
    
    if not user_id or not org_id:
        error_msg = "Error: user_id and org_id are required in config"
        logger.error(f"[TOOL_ERROR] search_row_values | Missing credentials")
        return [{"type": "text", "text": error_msg}]
    
    try:
        log_tool_call(
            "search_row_values",
            {
                "value_pattern": value_pattern,
                "column_name": column_name,
                "document_id": document_id,
                "limit": limit,
            },
            user_id,
            org_id,
        )

        writer = get_stream_writer()
        writer(f"search_row_values: {value_pattern}")
        
        results = await execute_search_row_values(
            user_id=user_id,
            org_id=org_id,
            value_pattern=value_pattern,
            column_name=column_name,
            document_id=document_id,
            limit=limit
        )
        
        
        formatted_results = await format_neo4j_results(results)
        log_tool_success("search_row_values", len(results), user_id, org_id)
        return formatted_results
    except Exception as e:
        log_tool_error("search_row_values", e, user_id, org_id)
        return [{"type": "text", "text": f"Error executing search_row_values: {str(e)}"}]

class QueryTabularDataSchema(BaseModel):
    document_id: str = Field(description="Document internal object ID")
    sheet_name: Optional[str] = Field(default=None, description="Filter by sheet name")
    row_index: Optional[int] = Field(default=None, description="Specific row to retrieve")

@tool(
    name_or_callable="query_tabular_data",
    description=get_tool_description("query_tabular_data"),
    args_schema=QueryTabularDataSchema,
    response_format="content"
)
async def query_tabular_data(
    document_id: str,
    sheet_name: Optional[str] = None,
    row_index: Optional[int] = None,
    config: RunnableConfig = None
) -> List[Dict[str, Any]]:
    user_id = config.get("configurable", {}).get("user_id")
    org_id = config.get("configurable", {}).get("org_id")
    
    if not user_id or not org_id:
        error_msg = "Error: user_id and org_id are required in config"
        logger.error(f"[TOOL_ERROR] query_tabular_data | Missing credentials")
        return [{"type": "text", "text": error_msg}]
    
    try:
        log_tool_call("query_tabular_data", {"document_id": document_id}, user_id, org_id)

        writer = get_stream_writer()
        writer(f"query_tabular_data: {document_id}, sheet={sheet_name}")
        
        results = await execute_query_tabular_data(
            user_id=user_id,
            org_id=org_id,
            document_id=document_id,
            sheet_name=sheet_name,
            row_index=row_index
        )
        
        
        formatted_results = await format_neo4j_results(results)
        log_tool_success("query_tabular_data", len(results), user_id, org_id)
        return formatted_results
    except Exception as e:
        log_tool_error("query_tabular_data", e, user_id, org_id)
        return [{"type": "text", "text": f"Error executing query_tabular_data: {str(e)}"}]

class HybridSearchSchema(BaseModel):
    query_text: str = Field(description="Natural language query")
    search_nodes: List[str] = Field(default=['Page', 'Entity', 'Column'], description="Node types to search")
    include_relationships: bool = Field(default=True, description="Expand to connected nodes")
    limit: int = Field(default=20, description="Max results per node type")

@tool(
    name_or_callable="hybrid_search",
    description=get_tool_description("hybrid_search"),
    args_schema=HybridSearchSchema,
    response_format="content"
)
async def hybrid_search(
    query_text: str,
    search_nodes: List[str] = ['Page', 'Entity', 'Column'],
    include_relationships: bool = True,
    limit: int = 20,
    config: RunnableConfig = None
) -> List[Dict[str, Any]]:
    user_id = config.get("configurable", {}).get("user_id")
    org_id = config.get("configurable", {}).get("org_id")
    
    if not user_id or not org_id:
        error_msg = "Error: user_id and org_id are required in config"
        logger.error(f"[TOOL_ERROR] hybrid_search | Missing credentials")
        return [{"type": "text", "text": error_msg}]
    
    try:
        log_tool_call("hybrid_search", {"query_text": query_text[:100], "search_nodes": search_nodes, "include_relationships": include_relationships, "limit": limit}, user_id, org_id)

        writer = get_stream_writer()
        writer(f"hybrid_search: {query_text[:50]}, nodes={search_nodes}")
        
        results = await execute_hybrid_search(
            user_id=user_id,
            org_id=org_id,
            query_text=query_text,
            search_nodes=search_nodes,
            include_relationships=include_relationships,
            limit=limit
        )
        
        
        formatted_results = await format_neo4j_results(results)
        log_tool_success("hybrid_search", len(results), user_id, org_id)
        return formatted_results
    except Exception as e:
        log_tool_error("hybrid_search", e, user_id, org_id)
        return [{"type": "text", "text": f"Error executing hybrid_search: {str(e)}"}]

class BreadthFirstSearchSchema(BaseModel):
    start_node_type: str = Field(description="'Document', 'Page', or 'Entity'")
    start_node_id: str = Field(description="Starting node identifier")
    max_depth: int = Field(default=2, description="BFS depth (1-3)")
    relationship_types: Optional[List[str]] = Field(default=None, description="Filter edge types")
    limit: int = Field(default=100, description="Max results")

@tool(
    name_or_callable="breadth_first_search",
    description=get_tool_description("breadth_first_search"),
    args_schema=BreadthFirstSearchSchema,
    response_format="content"
)
async def breadth_first_search(
    start_node_type: str,
    start_node_id: str,
    max_depth: int = 2,
    relationship_types: Optional[List[str]] = None,
    limit: int = 100,
    config: RunnableConfig = None
) -> List[Dict[str, Any]]:
    user_id = config.get("configurable", {}).get("user_id")
    org_id = config.get("configurable", {}).get("org_id")
    
    if not user_id or not org_id:
        error_msg = "Error: user_id and org_id are required in config"
        logger.error(f"[TOOL_ERROR] breadth_first_search | Missing credentials")
        return [{"type": "text", "text": error_msg}]
    
    try:
        log_tool_call("breadth_first_search", {"start_node_type": start_node_type, "start_node_id": start_node_id, "max_depth": max_depth, "relationship_types": relationship_types, "limit": limit}, user_id, org_id)

        writer = get_stream_writer()
        writer(f"breadth_first_search: {start_node_type}/{start_node_id}, depth={max_depth}")
        
        results = await execute_breadth_first_search(
            user_id=user_id,
            org_id=org_id,
            start_node_type=start_node_type,
            start_node_id=start_node_id,
            max_depth=max_depth,
            relationship_types=relationship_types,
            limit=limit
        )
        
        
        formatted_results = await format_neo4j_results(results)
        log_tool_success("breadth_first_search", len(results), user_id, org_id)
        return formatted_results
    except Exception as e:
        log_tool_error("breadth_first_search", e, user_id, org_id)
        return [{"type": "text", "text": f"Error executing breadth_first_search: {str(e)}"}]

class GetEntityContextSchema(BaseModel):
    entity_id: str = Field(description="Entity ID")
    document_id: str = Field(description="Document internal object ID")
    include_pages: bool = Field(default=True, description="Include pages mentioning entity")
    include_related_entities: bool = Field(default=True, description="Include related entities")
    include_document: bool = Field(default=True, description="Include document context")

@tool(
    name_or_callable="get_entity_context",
    description=get_tool_description("get_entity_context"),
    args_schema=GetEntityContextSchema,
    response_format="content"
)
async def get_entity_context(
    entity_id: str,
    document_id: str,
    include_pages: bool = True,
    include_related_entities: bool = True,
    include_document: bool = True,
    config: RunnableConfig = None
) -> List[Dict[str, Any]]:
    user_id = config.get("configurable", {}).get("user_id")
    org_id = config.get("configurable", {}).get("org_id")
    
    if not user_id or not org_id:
        error_msg = "Error: user_id and org_id are required in config"
        logger.error(f"[TOOL_ERROR] get_entity_context | Missing credentials")
        return [{"type": "text", "text": error_msg}]
    
    try:
        log_tool_call("get_entity_context", {"entity_id": entity_id, "document_id": document_id, "include_pages": include_pages, "include_related_entities": include_related_entities, "include_document": include_document}, user_id, org_id)

        writer = get_stream_writer()
        writer(f"get_entity_context: {entity_id}")
        
        results = await execute_get_entity_context(
            user_id=user_id,
            org_id=org_id,
            entity_id=entity_id,
            document_id=document_id,
            include_pages=include_pages,
            include_related_entities=include_related_entities,
            include_document=include_document
        )
        
        
        formatted_results = await format_neo4j_results(results)
        log_tool_success("get_entity_context", len(results), user_id, org_id)
        return formatted_results
    except Exception as e:
        log_tool_error("get_entity_context", e, user_id, org_id)
        return [{"type": "text", "text": f"Error executing get_entity_context: {str(e)}"}]

class RawCypherQuerySchema(BaseModel):
    cypher_query: str = Field(description="Cypher query to execute against the knowledge graph")
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Parameter map for the Cypher query")

@tool(
    name_or_callable="execute_raw_cypher_query",
    description=get_tool_description("execute_raw_cypher_query"),
    args_schema=RawCypherQuerySchema,
    response_format="content"
)
async def execute_raw_cypher_query(
    cypher_query: str,
    parameters: Optional[Dict[str, Any]] = None,
    config: RunnableConfig = None
) -> List[Dict[str, Any]]:
    user_id = config.get("configurable", {}).get("user_id")
    org_id = config.get("configurable", {}).get("org_id")
    if not user_id or not org_id:
        error_msg = "Error: user_id and org_id are required in config"
        logger.error(f"[TOOL_ERROR] execute_raw_cypher_query | Missing credentials")
        return [{"type": "text", "text": error_msg}]
    log_tool_call("execute_raw_cypher_query", {"cypher_query": cypher_query[:200], "parameters": parameters}, user_id, org_id)
    try:
        writer = get_stream_writer()
        writer(f"execute_raw_cypher_query: {cypher_query[:80]}")
        results = await execute_raw_cypher(
            user_id=user_id,
            org_id=org_id,
            cypher_query=cypher_query,
            parameters=parameters or {}
        )
        formatted_results = await format_neo4j_results(results)
        log_tool_success("execute_raw_cypher_query", len(results), user_id, org_id)
        return formatted_results
    except Exception as e:
        log_tool_error("execute_raw_cypher_query", e, user_id, org_id)
        return [{"type": "text", "text": f"Error executing execute_raw_cypher_query: {str(e)}"}]
