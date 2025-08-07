import aiohttp
import yaml
import json
import asyncio
from pathlib import Path
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from typing import Optional, Dict, Any, List, Union
from langgraph.config import get_stream_writer
import re
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
    


def validate_cypher_syntax(query: str) -> tuple[bool, str, str]:
    """
    Enhanced Cypher query validation with security and performance checks
    Returns: (is_valid, corrected_query, message)
    """
    corrected_query = query.strip()
    messages = []
    
    pattern = r'\{key:\s*key,\s*value:\s*d\[key\]\]'
    if re.search(pattern, corrected_query):
        corrected_query = re.sub(pattern, '{key: key, value: d[key]}]', corrected_query)
        messages.append("Auto-corrected metadata syntax (fixed missing closing brace)")
    
    open_braces = corrected_query.count('{')
    close_braces = corrected_query.count('}')
    open_brackets = corrected_query.count('[')
    close_brackets = corrected_query.count(']')
    open_parens = corrected_query.count('(')
    close_parens = corrected_query.count(')')
    
    if open_braces != close_braces:
        return False, corrected_query, f"Mismatched braces: {open_braces} open, {close_braces} close"
    
    if open_brackets != close_brackets:
        return False, corrected_query, f"Mismatched brackets: {open_brackets} open, {close_brackets} close"
        
    if open_parens != close_parens:
        return False, corrected_query, f"Mismatched parentheses: {open_parens} open, {close_parens} close"
    
    if '$user_id' not in corrected_query or '$org_id' not in corrected_query:
        return False, corrected_query, "Missing required security parameters ($user_id and $org_id)"
    
    dangerous_patterns = [
        (r'CALL\s+apoc\.', "APOC procedure calls not allowed"),
        (r'LOAD\s+CSV', "LOAD CSV operations not allowed"),
        (r'CALL\s+db\.', "Database procedure calls not allowed"),
        (r'CALL\s+dbms\.', "DBMS procedure calls not allowed"),
        (r'CREATE\s+INDEX', "Index creation not allowed"),
        (r'DROP\s+', "DROP operations not allowed"),
        (r'ALTER\s+', "ALTER operations not allowed"),
    ]
    
    query_upper = corrected_query.upper()
    for pattern, error_msg in dangerous_patterns:
        if re.search(pattern, query_upper):
            return False, corrected_query, f"Security violation: {error_msg}"
    
    valid_clauses = ['MATCH', 'OPTIONAL MATCH', 'CREATE', 'MERGE', 'DELETE', 'DETACH DELETE',
                     'SET', 'REMOVE', 'RETURN', 'WITH', 'WHERE', 'ORDER BY', 'SKIP', 'LIMIT',
                     'UNWIND', 'UNION', 'CALL']
    
    has_valid_clause = any(clause in query_upper for clause in valid_clauses)
    if not has_valid_clause:
        return False, corrected_query, "No valid Cypher clauses found"
    
    if 'LIMIT' not in query_upper:
        return False, corrected_query, "Missing required LIMIT clause"
    
    if re.search(r'MATCH\s*\([^)]*\)\s*(?!.*WHERE)', corrected_query, re.IGNORECASE):
        messages.append("Warning: MATCH without WHERE clause may cause performance issues")
    
    limit_match = re.search(r'LIMIT\s+(\d+)', query_upper)
    if limit_match:
        limit_value = int(limit_match.group(1))
        if limit_value > 1000:
            return False, corrected_query, f"LIMIT value too high: {limit_value} (max allowed: 1000)"
        if limit_value <= 0:
            return False, corrected_query, f"LIMIT value must be positive: {limit_value}"
    
    node_pattern = r'\([^)]*:[^)]+\)'  # (n:Label) pattern
    relationship_pattern = r'-\[[^\]]*:[^\]]+\]-'  # -[r:TYPE]- pattern
    
    if 'MATCH' in query_upper and not re.search(node_pattern, corrected_query):
        messages.append("Warning: MATCH clause should typically include node labels")
    
    invalid_params = re.findall(r'\$[a-zA-Z_]\w*', corrected_query)
    required_params = {'$user_id', '$org_id'}
    for param in invalid_params:
        if param not in required_params and not re.match(r'^\$[a-zA-Z_][a-zA-Z0-9_]*$', param):
            return False, corrected_query, f"Invalid parameter syntax: {param}"
    
    if ';' in corrected_query.rstrip(';'):
        return False, corrected_query, "Multiple statements not allowed (semicolon found)"
    
    return True, corrected_query, "; ".join(messages) if messages else "Query validated successfully"


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

async def process_single_record(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    content_found = False
    final_content = []
    metadata_parts = []
    
    for key, value in record.items():
        if "content" in key and isinstance(value, str):
            content_items = format_content_field(value)
            final_content.extend(content_items)
            content_found = True
        elif isinstance(value, dict):
            content_key_found = None
            for nested_key in value.keys():
                if "content" in nested_key and isinstance(value[nested_key], str):
                    content_key_found = nested_key
                    break
            
            if content_key_found:
                content_items = format_content_field(value[content_key_found])
                final_content.extend(content_items)
                content_found = True
                
                for nested_key, nested_value in value.items():
                    if nested_key != content_key_found and nested_value is not None:
                        metadata_parts.append(f"{nested_key}: {nested_value}")
        else:
            if isinstance(value, dict):
                nested_parts = []
                for nested_key, nested_value in value.items():
                    if nested_value is not None:
                        nested_parts.append(f"{nested_key}: {nested_value}")
                if nested_parts:
                    metadata_parts.append(f"{key}: {{{', '.join(nested_parts)}}}")
            elif value is not None:
                metadata_parts.append(f"{key}: {value}")
    
    if content_found and metadata_parts:
        final_content.append({"type": "text", "text": f"[METADATA] {' | '.join(metadata_parts)}"})
    elif not content_found:
        final_content.append({"type": "text", "text": json.dumps(record, indent=2)})
    
    return final_content

async def format_neo4j_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not results:
        return []
    
    processed_records = await asyncio.gather(*[process_single_record(record) for record in results])
    
    formatted_results = []
    for record_content in processed_records:
        formatted_results.extend(record_content)
    
    return formatted_results

@tool(
    name_or_callable=TOOL_NAME,
    description=get_tool_description(TOOL_NAME),
    args_schema=KnowledgeSearch,
    parse_docstring=False,
    infer_schema=True,
    response_format="content"
)
async def knowledge_search_tool(query: str, parameters: Optional[Dict[str, Any]] = None, text_to_embed: Optional[str] = None, config: RunnableConfig = None) -> List[Dict[str, Any]]:
    url = f"{app_config.INDEXER_API_BASE_URL}/search/cypher"
    
    user_id = config.get("configurable", {}).get("user_id")
    org_id = config.get("configurable", {}).get("org_id")
    
    if not user_id or not org_id:
        return [{"type": "text", "text": "Error: user_id and org_id are required in config"}]
    
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
        
        is_valid, corrected_query, validation_message = validate_cypher_syntax(query)
        
        if not is_valid:
            return [{"type": "text", "text": f"Query validation error: {validation_message}\nOriginal query: {query}"}]
        
        if corrected_query != query:
            writer(f"#### Query Auto-Corrected #### {validation_message}")
            query = corrected_query
        
        parameters = parameters or {}
        
        if text_to_embed and "$embedding" in query:
            embedding = await generate_embedding(text_to_embed)
            parameters["embedding"] = embedding
        elif "$embedding" in query and not text_to_embed:
            return [{"type": "text", "text": "Error: Query contains $embedding parameter but no text_to_embed provided"}]
        
        query, parameters = validate_and_inject_security_params(query, parameters)
        
        payload = {
            "query": query,
            "parameters": parameters
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    formatted_result = await format_neo4j_results(result)
                    return formatted_result
                else:
                    return [{"type": "text", "text": f"Search API error: {response.status} - {await response.text()}"}]
    except ValueError as e:
        return [{"type": "text", "text": f"Security error: {str(e)}"}]
    except Exception as e:
        return [{"type": "text", "text": f"Query processing error: {str(e)}"}]