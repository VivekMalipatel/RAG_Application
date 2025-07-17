import aiohttp
from aiohttp import ClientSession
import asyncio
import yaml
import json
from pathlib import Path
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from typing import Optional, Dict, Any, List, Union

from pydantic import BaseModel, Field
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

class KnowledgeSearchRequest(BaseModel):
    requests: List[KnowledgeSearch] = Field(
        description="List of knowledge search requests to execute concurrently (max 10 requests)."
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

def format_neo4j_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    formatted_results = []
    
    for record in results:
        final_content = []
        metadata_parts = []
        
        for key, value in record.items():
            if key == "content" and isinstance(value, str):
                content_items = format_content_field(value)
                final_content.extend(content_items)
            elif isinstance(value, dict) and "content" in value and isinstance(value["content"], str):
                content_items = format_content_field(value["content"])
                final_content.extend(content_items)
            elif key != "content":
                if isinstance(value, dict):
                    nested_parts = []
                    for nested_key, nested_value in value.items():
                        if nested_key != "content":
                            nested_parts.append(f"{nested_key}: {nested_value}")
                    if nested_parts:
                        metadata_parts.append(f"{key}: {{{', '.join(nested_parts)}}}")
                elif value is not None:
                    metadata_parts.append(f"{key}: {value}")
        
        if metadata_parts:
            final_content.append({"type": "text", "text": f"[METADATA] {' | '.join(metadata_parts)}"})
        
        if not final_content:
            final_content.append({"type": "text", "text": json.dumps(record, indent=2)})
        
        formatted_results.append({"content": final_content})
    
    return formatted_results

@tool(
    name_or_callable=TOOL_NAME,
    description=get_tool_description(TOOL_NAME),
    args_schema=KnowledgeSearchRequest,
    parse_docstring=False,
    infer_schema=True,
    response_format="content"
)
async def knowledge_search_tool(requests: List[KnowledgeSearch], config: RunnableConfig) -> List[str]:
    url = f"{app_config.INDEXER_API_BASE_URL}/search/cypher"
    
    user_id = config.get("configurable", {}).get("user_id")
    org_id = config.get("configurable", {}).get("org_id")
    
    if not user_id or not org_id:
        return ["Error: user_id and org_id are required in config"]
    
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
        parameters["user_id"] = user_id
        parameters["org_id"] = org_id
        
        return query, parameters
    
    async def fetch(session: ClientSession, payload):
        try:
            query = payload.get("query", "")
            parameters = payload.get("parameters", {})
            text_to_embed = payload.get("text_to_embed")
            
            if text_to_embed and "$embedding" in query:
                embedding = await generate_embedding(text_to_embed)
                parameters["embedding"] = embedding
            elif "$embedding" in query and not text_to_embed:
                return "Error: Query contains $embedding parameter but no text_to_embed provided"
            
            query, parameters = validate_and_inject_security_params(query, parameters)
            
            validated_payload = {
                "query": query,
                "parameters": parameters
            }
            
            async with session.post(url, json=validated_payload) as response:
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

    async with aiohttp.ClientSession() as session:
        tasks = []
        for req in requests:
            payload = {
                "query": req.query,
                "parameters": req.parameters or {},
                "text_to_embed": req.text_to_embed
            }
            tasks.append(fetch(session, payload))
        return await asyncio.gather(*tasks)