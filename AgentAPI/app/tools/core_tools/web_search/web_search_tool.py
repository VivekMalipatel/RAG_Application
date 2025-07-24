import yaml
import json
from pathlib import Path
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from langchain_community.utilities import SearxSearchWrapper
from config import config as app_config

TOOL_NAME = "web_search_tool"

class WebSearch(BaseModel):
    query: str = Field(
        description="Web search query to execute. Use clear, specific terms for best results."
    )
    num_results: Optional[int] = Field(
        default=10,
        description="Number of results to return (default: 10, max: 50)."
    )
    language: Optional[str] = Field(
        default="en",
        description="Language preference for search results."
    )
    engines: Optional[List[str]] = Field(
        default=None,
        description="List of search engines to use (e.g., ['google', 'bing'])."
    )
    categories: Optional[List[str]] = Field(
        default=None,
        description="List of categories to use (e.g., ['general', 'news'])."
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

def format_web_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
    args_schema=WebSearch,
    parse_docstring=False,
    infer_schema=True,
    response_format="content"
)
async def web_search_tool(query: str, num_results: Optional[int] = 10, language: Optional[str] = "en", engines: Optional[List[str]] = None, categories: Optional[List[str]] = None, config: RunnableConfig = None) -> str:
    try:
        searx = SearxSearchWrapper(
            searx_host=f"{app_config.SEARX_URL}",
            engines=engines,
            categories=categories,
            params={
                "language": language or "en",
                "num_results": min(num_results or 10, 50)
            }
        )
        search_results = searx.run(query)
        # If Searx returns a string, wrap it in a list of dicts for formatting
        if isinstance(search_results, str):
            search_results = [{"content": search_results}]
        formatted_result = format_web_results(search_results)
        return json.dumps(formatted_result, indent=2)
    except Exception as e:
        error_result = [{"content": [{"type": "text", "text": f"Search error: {str(e)}"}]}]
        return json.dumps(error_result, indent=2)

# Example usage for local testing
if __name__ == "__main__":
    query = "LangChain Python documentation"
    num_results = 5
    language = "en"
    engines = ["google"]
    categories = ["general"]
    print("Web Search Test Result:")
    try:
        searx = SearxSearchWrapper(
            searx_host=f"{app_config.SEARX_URL}",
            engines=engines,
            categories=categories,
            params={
                "language": language,
                "num_results": num_results
            }
        )
        search_results = searx.run(query)
        if isinstance(search_results, str):
            search_results = [{"content": search_results}]
        formatted_result = format_web_results(search_results)
        print(json.dumps(formatted_result, indent=2))
    except Exception as e:
        print(json.dumps([{"content": [{"type": "text", "text": f"Search error: {str(e)}"}]}], indent=2))