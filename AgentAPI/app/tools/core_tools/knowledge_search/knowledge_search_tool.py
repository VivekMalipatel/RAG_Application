import aiohttp
import yaml
import json
from pathlib import Path
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from typing import Optional, Dict, Any, List, Union
from langgraph.config import get_stream_writer

from pydantic import Field, BaseModel
from langchain_openai import OpenAIEmbeddings
from config import config as app_config

TOOL_NAME = "knowledge_search_tool"

class KnowledgeSearch(BaseModel):
    query: str = Field(
        description="Cypher query to execute against the knowledge graph. Must include $user_id and $org_id parameters for security filtering."
    )
    parameters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional parameters for the Cypher query. user_id and org_id will be automatically injected."
    )
    text_to_embed: Optional[str] = Field(
        default=None,
        description="Text to convert to embedding vector. Required when query contains $embedding parameter for vector similarity search."
    )

def get_tool_description(tool_name: str, yaml_filename: str = "description.yaml") -> str:
    yaml_path = Path(__file__).parent / yaml_filename
    with open(yaml_path, 'r') as f:
        descriptions = yaml.safe_load(f)
    return descriptions.get(tool_name, "")

def format_content_field(content_str: str) -> List[Dict[str, Any]]:
    try:
        content_data = json.loads(content_str)
        if isinstance(content_data, list):
            formatted_content = []
            for item in content_data:
                if isinstance(item, dict):
                    if item.get("type") == "image_url" and "image_url" in item:
                        formatted_content.append({
                            "type": "image_url",
                            "image_url": item["image_url"]
                        })
                    elif item.get("type") == "text" and "text" in item:
                        formatted_content.append({
                            "type": "text", 
                            "text": item["text"]
                        })
            return formatted_content
        else:
            return [{"type": "text", "text": str(content_data)}]
    except (json.JSONDecodeError, TypeError):
        return [{"type": "text", "text": str(content_str)}]

def format_neo4j_results(results: List[Dict[str, Any]], query_type: str = "default") -> List[Dict[str, Any]]:
    """
    Format Neo4j results for user-friendly presentation
    
    Args:
        results: Raw Neo4j query results
        query_type: Type of query to customize formatting ("documents", "entities", "content", etc.)
    """
    formatted_results = []
    
    HIDDEN_FIELDS = {
        'internal_object_id', 'task_id', 'user_id', 'org_id', 's3_url',
        'embedding', 'id' 
    }
    
    SKIP_PLACEHOLDER_METADATA = {
        'metadata_1', 'metadata_2', 'metadata_3', 'metadata_4', 'metadata_5'
    }
    
    for record in results:
        final_content = []
        
        if query_type == "documents":
            formatted_record = format_document_record(record)
        elif query_type == "entities":
            formatted_record = format_entity_record(record)
        elif query_type == "content":
            formatted_record = format_content_record(record)
        else:
            formatted_record = format_generic_record(record, HIDDEN_FIELDS, SKIP_PLACEHOLDER_METADATA)
        
        if formatted_record:
            final_content.append({"type": "text", "text": formatted_record})
        
        if not final_content:
            final_content.append({"type": "text", "text": "No readable content found in this record"})
        
        formatted_results.append({"content": final_content})
    
    return formatted_results

def format_document_record(record: Dict[str, Any]) -> str:
    """Format document records for clean user presentation"""
    filename = clean_filename(record.get('filename', 'Unknown Document'))
    file_type = record.get('file_type', 'Unknown')
    category = record.get('category', 'Uncategorized')
    source = record.get('source', 'Unknown Source')
    
    doc_info = f"**{filename}**"
    if file_type != 'Unknown':
        doc_info += f" ({file_type})"
    
    details = []
    if category != 'Uncategorized':
        details.append(f"Category: {category}")
    if source != 'Unknown Source':
        details.append(f"Source: {source.replace('-', ' ').title()}")
    
    if details:
        doc_info += f"\n   • {' | '.join(details)}"
    
    return doc_info

def format_entity_record(record: Dict[str, Any]) -> str:
    """Format entity records for clean user presentation"""
    entity_text = record.get('text', 'Unknown Entity')
    entity_type = record.get('entity_type', 'Unknown Type')
    entity_profile = record.get('entity_profile', '')
    
    entity_info = f"🔍 **{entity_text}** ({entity_type})"
    
    if entity_profile and len(entity_profile) > 10: 
        profile_preview = entity_profile[:100] + "..." if len(entity_profile) > 100 else entity_profile
        entity_info += f"\n   📝 {profile_preview}"
    
    return entity_info

def format_content_record(record: Dict[str, Any]) -> str:
    """Format content records (pages, etc.) for clean user presentation"""
    content_parts = []
    
    if 'content' in record and isinstance(record['content'], str):
        try:
            content_data = json.loads(record['content'])
            if isinstance(content_data, list):
                for item in content_data:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text = item.get("text", "")
                        if text and len(text.strip()) > 10:
                            # Truncate long content for preview
                            preview = text[:200] + "..." if len(text) > 200 else text
                            content_parts.append(preview)
        except (json.JSONDecodeError, TypeError):
            content_text = str(record['content'])[:200]
            if content_text.strip():
                content_parts.append(content_text)
    
    page_info = ""
    if 'page_number' in record:
        page_info = f"📄 Page {record['page_number']}: "
    elif 'filename' in record:
        filename = clean_filename(record['filename'])
        page_info = f"📄 From {filename}: "
    
    if content_parts:
        return page_info + "\n".join(content_parts)
    else:
        return page_info + "Content available but not displayable in preview"

def format_generic_record(record: Dict[str, Any], hidden_fields: set, skip_metadata: set) -> str:
    """Format any other type of record"""
    visible_parts = []
    
    for key, value in record.items():
        if key.lower() in hidden_fields:
            continue
            
        if key in skip_metadata and (value == "[value]" or not value):
            continue
        
        if key == "content" and isinstance(value, str):
            try:
                content_data = json.loads(value)
                if isinstance(content_data, list):
                    text_content = []
                    for item in content_data:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text_content.append(item.get("text", ""))
                    if text_content:
                        content_preview = " ".join(text_content)[:150] + "..."
                        visible_parts.append(f"Content: {content_preview}")
            except (json.JSONDecodeError, TypeError):
                if len(str(value)) > 10:  # Only show meaningful content
                    visible_parts.append(f"Content: {str(value)[:150]}...")
        
        elif value is not None and str(value).strip():
            if isinstance(value, dict):
                nested_parts = []
                for nested_key, nested_value in value.items():
                    if nested_key != "content" and nested_value:
                        nested_parts.append(f"{nested_key}: {nested_value}")
                if nested_parts:
                    visible_parts.append(f"{key}: {{{', '.join(nested_parts)}}}")
            else:
                clean_value = clean_field_value(str(value))
                if clean_value:
                    visible_parts.append(f"{key}: {clean_value}")
    
    return " | ".join(visible_parts) if visible_parts else "No displayable information"

def clean_filename(filename: str) -> str:
    """Clean up filenames by removing UUIDs and technical prefixes"""
    if not filename:
        return "Unknown Document"
    
    import re
    cleaned = re.sub(r'^[a-f0-9\-]{36}_', '', filename)
    cleaned = re.sub(r'^\d+_[a-f0-9\-]+_[^_]+_', '', cleaned)
    
    return cleaned or filename  # Return original if cleaning removes everything

def clean_field_value(value: str) -> str:
    """Clean up field values for display"""
    if value in ["[value]", "", "null", "None"]:
        return ""
    
    if "open-webui" in value.lower():
        return "Open WebUI"
    
    return value



@tool(
    name_or_callable=TOOL_NAME,
    description=get_tool_description(TOOL_NAME),
    args_schema=KnowledgeSearch,
    parse_docstring=False,
    infer_schema=True,
    response_format="content"
)
async def knowledge_search_tool(query: str, parameters: Optional[Dict[str, Any]] = None, text_to_embed: Optional[str] = None, config: RunnableConfig = None) -> str:
    url = f"{app_config.INDEXER_API_BASE_URL}/search/cypher"
    
    user_id = config.get("configurable", {}).get("user_id")
    org_id = config.get("configurable", {}).get("org_id")
    
    if not user_id or not org_id:
        return "Error: user_id and org_id are required in config"
    
    embeddings_client = OpenAIEmbeddings(
        model=app_config.MULTIMODEL_EMBEDDING_MODEL,
        base_url=app_config.OPENAI_BASE_URL,
        api_key=app_config.OPENAI_API_KEY
    )
    
    async def generate_embedding(text: str) -> List[float]:
        try:
            embedding = await embeddings_client.aembed_query(text)
            return embedding
        except Exception as e:
            raise ValueError(f"Failed to generate embedding: {str(e)}")
    
    def validate_and_inject_security_params(query: str, parameters: dict) -> tuple[str, dict]:
        if "$user_id" not in query or "$org_id" not in query:
            raise ValueError("Security violation: All Cypher queries must include $user_id and $org_id parameters")
        
        parameters = parameters or {}
        parameters["user_id"] = str(user_id).split('$')[0]
        parameters["org_id"] = str(org_id).split('$')[0]
        
        return query, parameters
    
    try:
        writer = get_stream_writer()
        writer(f"Knowledge Search Query: {query}")
        
        parameters = parameters or {}
        
        if text_to_embed and "$embedding" in query:
            embedding = await generate_embedding(text_to_embed)
            parameters["embedding"] = embedding
        elif "$embedding" in query and not text_to_embed:
            return "Error: Query contains $embedding parameter but no text_to_embed provided"
        
        query, parameters = validate_and_inject_security_params(query, parameters)
        
        payload = {
            "query": query,
            "parameters": parameters
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    formatted_result = format_neo4j_results(result)
                    return json.dumps(formatted_result, indent=2)
                else:
                    return f"Search API error: {response.status} - {await response.text()}"
    except ValueError as e:
        return f"Security error: {str(e)}"
    except Exception as e:
        return f"Query processing error: {str(e)}"