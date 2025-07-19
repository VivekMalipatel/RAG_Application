from scrapegraphai.graphs import SmartScraperMultiGraph
import os
from scrapegraphai.graphs import SmartScraperGraph
from dotenv import load_dotenv
import json
import re
from pydantic import BaseModel, Field
from typing import Optional, Any, List
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
load_dotenv()

llm_config = {
    "base_url": os.getenv("OPENAI_API_BASE"),
    "api_key": os.getenv("OPENAI_API_KEY"),
    "model": os.getenv("WEB_SCRAPE_MODEL"),
    "temperature": 0.0
}
TOOL_NAME = "web_scrape_tool"
class ScrapedContent(BaseModel):
    content: Any

class WebScrapeRequest(BaseModel):
    url: str = Field(description="URL to scrape")
    extraction_prompt: str = Field(description="Natural language description of what to extract from the webpage")

class MultiWebScrapeRequest(BaseModel):
    urls: List[str] = Field(description="List of URLs to scrape")
    extraction_prompt: str = Field(description="Natural language description of what to extract from the webpages")

def clean_json_output(text: str) -> str:
    """Clean malformed JSON output from LLM"""
    # Remove double braces
    text = re.sub(r'\{\{', '{', text)
    text = re.sub(r'\}\}', '}', text)
    return text


def scrape_multiple_websites(prompt: str, urls: list[str]) -> str:
    """
    Scrape multiple websites using ScrapeGraphAI's SmartScraperMultiGraph.
    Args:
        prompt: Natural language description of what to extract
        urls: List of URLs to scrape
    Returns:
        Extracted data as a string
    """
    graph_config = {
        "llm": llm_config,
        "verbose": True,
        "headless": True,
        "loader_kwargs": {
            "channel": "chrome",
        }
    }
    try:
        multi_scraper = SmartScraperMultiGraph(
            prompt=prompt,
            source=urls,
            config=graph_config,
            schema=ScrapedContent
        )
        result = multi_scraper.run()
        print(f"Multi-scrape result: {result}")
        
        # Handle the result properly
        if isinstance(result, dict):
            return json.dumps(result, ensure_ascii=False)
        elif isinstance(result, str):
            cleaned_result = clean_json_output(result)
            try:
                parsed_result = json.loads(cleaned_result)
                return json.dumps(parsed_result, ensure_ascii=False)
            except json.JSONDecodeError:
                return json.dumps({"content": cleaned_result}, ensure_ascii=False)
        else:
            return json.dumps({"content": str(result)}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"Error multi-scraping {urls}: {str(e)}"}, ensure_ascii=False)

def scrape_website(url: str, extraction_prompt: str) -> str:
    """
    Scrape a website using ScrapeGraphAI with natural language instructions.
    Uses existing Chrome via Playwright to avoid lengthy downloads.
    """
    graph_config = {
        "llm": llm_config,
        "verbose": True,
        "headless": True,
        "loader_kwargs": {
            "channel": "chrome",
        }
    }

    try:
        scraper = SmartScraperGraph(
            prompt=extraction_prompt,
            source=url,
            config=graph_config,
            schema=ScrapedContent
        )
        result = scraper.run()
        
        # Handle the result properly
        if isinstance(result, dict):
            return json.dumps({"content": result}, ensure_ascii=False)
        elif isinstance(result, str):
            # Try to clean and parse if it's a string
            cleaned_result = clean_json_output(result)
            try:
                parsed_result = json.loads(cleaned_result)
                return json.dumps({"content": parsed_result}, ensure_ascii=False)
            except json.JSONDecodeError:
                return json.dumps({"content": cleaned_result}, ensure_ascii=False)
        else:
            return json.dumps({"content": str(result)}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)

@tool(
    name_or_callable=TOOL_NAME,
    description="Scrape content from websites using natural language instructions. Can scrape single or multiple URLs.",
    args_schema=WebScrapeRequest,
    parse_docstring=False,
    infer_schema=True,
    response_format="content"
)
async def web_scrape_tool(url: str, extraction_prompt: str, config: RunnableConfig) -> str:
    """
    Tool wrapper for web scraping that integrates with LangChain agents.
    """
    return scrape_website(url, extraction_prompt)

@tool(
    name_or_callable=TOOL_NAME,
    description="Scrape content from multiple websites using natural language instructions.",
    args_schema=MultiWebScrapeRequest,
    parse_docstring=False,
    infer_schema=True,
    response_format="content"
)
async def multi_web_scrape_tool(urls: List[str], extraction_prompt: str, config: RunnableConfig) -> str:
    """
    Tool wrapper for multi-website scraping that integrates with LangChain agents.
    """
    return scrape_multiple_websites(extraction_prompt, urls)

if __name__ == "__main__":
    # # Single-site scrape
    url = "https://www.bytecrafts.in/"
    extraction_prompt = "List all services provided by Bytecrafts."
    print("Scraped result:")
    print(scrape_website(url, extraction_prompt))

    # # Multi-site scrape
    multi_prompt = "Who is Marco Perini?"
    urls = [
        "https://perinim.github.io/",
        "https://perinim.github.io/cv/"
    ]
    print("\nMulti-site scraped result:")
    print(scrape_multiple_websites(multi_prompt, urls))
 