
# Refactored to match knowledge_search_tool.py style
import os
import json
import re
import asyncio
from pathlib import Path
from typing import Optional, Any, List, Dict
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from scrapegraphai.graphs import SmartScraperGraph, SmartScraperMultiGraph
from config import config as app_config
from langgraph.config import get_stream_writer


TOOL_NAME = "web_scrape_tool"

class WebScrapeTool(BaseModel):
    url: Optional[str] = Field(default=None, description="URL to scrape (for single site)")
    urls: Optional[List[str]] = Field(default=None, description="List of URLs to scrape (for multi-site)")
    extraction_prompt: str = Field(description="Natural language description of what to extract from the webpage(s)")

def get_tool_description(tool_name: str, yaml_filename: str = "description.yaml") -> str:
    yaml_path = Path(__file__).parent / yaml_filename
    if yaml_path.exists():
        import yaml
        with open(yaml_path, 'r') as f:
            descriptions = yaml.safe_load(f)
        return descriptions.get(tool_name, "")
    return "Scrape content from websites using natural language instructions. Can scrape single or multiple URLs."

def clean_json_output(text: str) -> str:
    text = re.sub(r'\{\{', '{', text)
    text = re.sub(r'\}\}', '}', text)
    return text

def format_scrape_result(result: Any) -> List[Dict[str, Any]]:
    formatted = []
    if isinstance(result, dict):
        for k, v in result.items():
            formatted.append({"type": "text", "text": f"{k}: {v}"})
    elif isinstance(result, list):
        for item in result:
            if isinstance(item, dict):
                for k, v in item.items():
                    formatted.append({"type": "text", "text": f"{k}: {v}"})
            else:
                formatted.append({"type": "text", "text": str(item)})
    elif isinstance(result, str):
        formatted.append({"type": "text", "text": result})
    else:
        formatted.append({"type": "text", "text": str(result)})
    return [{"content": formatted}]



def get_graph_config() -> dict:
    base_url = getattr(app_config, "OPENAI_BASE_URL", None)
    api_key = getattr(app_config, "OPENAI_API_KEY", None)
    # Try to get WEB_SCRAPE_MODEL from config, else from env
    model = getattr(app_config, "WEB_SCRAPE_MODEL", None) or os.getenv("WEB_SCRAPE_MODEL")
    missing = []
    if not base_url:
        missing.append("OPENAI_BASE_URL")
    if not api_key:
        missing.append("OPENAI_API_KEY")
    if not model:
        missing.append("WEB_SCRAPE_MODEL")
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
    llm_config = {
        "base_url": base_url,
        "api_key": api_key,
        "model": model,
        "temperature": 0.0
    }
    return {
        "llm": llm_config,
        "verbose": True,
        "headless": True,
        "loader_kwargs": {
            "channel": "chrome",
        }
    }

async def scrape_website(url: str, extraction_prompt: str) -> Any:
    try:
        graph_config = get_graph_config()
        scraper = SmartScraperGraph(
            prompt=extraction_prompt,
            source=url,
            config=graph_config
        )
        result = await scraper.run() if asyncio.iscoroutine(scraper.run()) else scraper.run()
        if isinstance(result, str):
            cleaned = clean_json_output(result)
            try:
                result = json.loads(cleaned)
            except Exception:
                result = cleaned
        return result
    except Exception as e:
        return {"error": str(e)}


def scrape_multiple_websites(extraction_prompt: str, urls: List[str]) -> Any:
    try:
        graph_config = get_graph_config()
        multi_scraper = SmartScraperMultiGraph(
            prompt=extraction_prompt,
            source=urls,
            config=graph_config
        )
        result = multi_scraper.run()
        if isinstance(result, str):
            cleaned = clean_json_output(result)
            try:
                result = json.loads(cleaned)
            except Exception:
                result = cleaned
        return result
    except Exception as e:
        return {"error": f"Error multi-scraping {urls}: {str(e)}"}

@tool(
    name_or_callable=TOOL_NAME,
    description=get_tool_description(TOOL_NAME),
    args_schema=WebScrapeTool,
    parse_docstring=False,
    infer_schema=True,
    response_format="content"
)
async def web_scrape_tool(url: Optional[str] = None, urls: Optional[List[str]] = None, extraction_prompt: str = "", config: RunnableConfig = None) -> str:
    writer = get_stream_writer()
    if urls:
        writer(f"#### Web Scrape #### URLs: {urls}\nPrompt: {extraction_prompt}")
        result = await scrape_multiple_websites(extraction_prompt, urls)
    elif url:
        writer(f"#### Web Scrape #### URL: {url}\nPrompt: {extraction_prompt}")
        result = await scrape_website(url, extraction_prompt)
    else:
        return json.dumps({"error": "No URL(s) provided"}, ensure_ascii=False)
    formatted = format_scrape_result(result)
    return json.dumps(formatted, indent=2, ensure_ascii=False)

# Example usage
if __name__ == "__main__":

    multi_prompt = "Extract the names and prices of shoes displayed on this page. Include currency and any sale prices. Format as: Product Name: $Price"
    urls = [
        "https://www.nike.com/us/w/mens-running-shoes-37v7jzy7ok",
        "https://www.nike.com/au/w/mens-running-shoes-37v7jzy7ok"
    ]
    print("\nMulti-site scraped result:")
    result_multi = scrape_multiple_websites(multi_prompt, urls)
    print(json.dumps(format_scrape_result(result_multi), indent=2, ensure_ascii=False))
 