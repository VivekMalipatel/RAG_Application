

import yaml
import json
from pathlib import Path
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from langchain_community.utilities import SearxSearchWrapper

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

class WebSearchRequest(BaseModel):
    requests: List[WebSearch] = Field(
        description="List of web search requests to execute concurrently (max 10 requests)."
    )

def get_tool_description(tool_name: str, yaml_filename: str = "description.yaml") -> str:
    yaml_path = Path(__file__).parent / yaml_filename
    with open(yaml_path, 'r') as f:
        descriptions = yaml.safe_load(f)
    return descriptions.get(tool_name, "")

def format_web_results(results) -> List[Dict[str, Any]]:
    formatted_results = []
    
    # Handle string results from SearxSearchWrapper
    if isinstance(results, str):
        # Detect HTML response
        if results.strip().lower().startswith("<html") or "<body" in results.lower():
            content_items = [{
                "type": "text",
                "text": "Web search API returned HTML instead of results. The SearX instance may be misconfigured or the query is unsupported."
            }]
        else:
            content_items = [{"type": "text", "text": results}]
        formatted_results.append({"content": content_items})
        return formatted_results
    
    # Handle list of dictionaries
    if isinstance(results, list):
        for result in results:
            if isinstance(result, dict):
                content_items = []
                # Title
                if result.get("title"):
                    content_items.append({"type": "text", "text": f"**{result['title']}**"})
                # Snippet/Description
                if result.get("snippet"):
                    content_items.append({"type": "text", "text": result["snippet"]})
                elif result.get("description"):
                    content_items.append({"type": "text", "text": result["description"]})
                # URL
                if result.get("url"):
                    content_items.append({"type": "text", "text": f"URL: {result['url']}"})
                elif result.get("link"):
                    content_items.append({"type": "text", "text": f"URL: {result['link']}"})
                # Source
                if result.get("source"):
                    content_items.append({"type": "text", "text": f"Source: {result['source']}"})
                # Fallback
                if not content_items:
                    content_items.append({"type": "text", "text": json.dumps(result, indent=2)})
                formatted_results.append({"content": content_items})
            else:
                # Handle non-dict items in list
                content_items = [{"type": "text", "text": str(result)}]
                formatted_results.append({"content": content_items})
        return formatted_results
    
    # Fallback for other types
    content_items = [{"type": "text", "text": str(results)}]
    formatted_results.append({"content": content_items})
    return formatted_results


@tool(
    name_or_callable=TOOL_NAME,
    description=get_tool_description(TOOL_NAME),
    args_schema=WebSearchRequest,
    parse_docstring=False,
    infer_schema=True,
    response_format="content"
)
async def web_search_tool(requests: List[WebSearch], config: RunnableConfig) -> List[str]:
    """
    Batch web search tool that executes multiple queries concurrently and returns formatted results using SearxSearchWrapper.
    """
    results = []
    for req in requests:
        try:
            # Configure SearxSearchWrapper for each request
            searx = SearxSearchWrapper(
                searx_host="https://websearch.gauravshivaprasad.com",
                engines=req.engines,
                categories=req.categories,
                params={
                    "language": req.language or "en",
                    "num_results": min(req.num_results or 10, 50)
                }
            )
            
            # Run the search
            search_results = searx.run(req.query)
            
            # Format results
            formatted_result = format_web_results(search_results)
            results.append(json.dumps(formatted_result, indent=2))
            
        except Exception as e:
            error_result = [{"content": [{"type": "text", "text": f"Search error: {str(e)}"}]}]
            results.append(json.dumps(error_result, indent=2))
    
    return results