"""Web-based tools for agents"""
import asyncio
import aiohttp
import json
from typing import Dict, Any, Optional
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field
from app.tools.base import BaseTool, ToolInput, ToolOutput
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

class WebSearchInput(ToolInput):
    """Input for web search tool"""
    query: str = Field(..., description="The search query")
    num_results: int = Field(default=5, description="Number of results to return")

class WebScrapingInput(ToolInput):
    """Input for web scraping tool"""
    url: str = Field(..., description="URL to scrape")
    extract_text: bool = Field(default=True, description="Whether to extract text content")
    extract_links: bool = Field(default=False, description="Whether to extract links")

class WebSearchTool(BaseTool):
    """Tool for searching the web"""
    
    def __init__(self):
        super().__init__(
            name="web_search",
            description="Search the web for information using DuckDuckGo"
        )
    
    async def execute(self, input_data: WebSearchInput) -> ToolOutput:
        """Execute web search"""
        try:
            # Using DuckDuckGo instant answer API with configuration
            url = settings.DUCKDUCKGO_API_URL + "/"
            params = {
                "q": input_data.query,
                "format": "json",
                "no_html": "1",
                "skip_disambig": "1"
            }
            
            timeout = aiohttp.ClientTimeout(total=settings.TOOL_API_TIMEOUT)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Extract relevant information
                        results = []
                        
                        # Abstract (instant answer)
                        if data.get("Abstract"):
                            results.append({
                                "type": "abstract",
                                "content": data["Abstract"],
                                "source": data.get("AbstractSource", "")
                            })
                        
                        # Related topics
                        for topic in data.get("RelatedTopics", [])[:input_data.num_results]:
                            if isinstance(topic, dict) and topic.get("Text"):
                                results.append({
                                    "type": "related",
                                    "content": topic["Text"],
                                    "url": topic.get("FirstURL", "")
                                })
                        
                        return ToolOutput(success=True, result={
                            "query": input_data.query,
                            "results": results
                        })
                    else:
                        return ToolOutput(
                            success=False,
                            error=f"Search failed with status {response.status}"
                        )
        
        except Exception as e:
            logger.error(f"Web search error: {str(e)}")
            return ToolOutput(success=False, error=str(e))
    
    def get_parameters_schema(self) -> Dict[str, Any]:
        """Get parameters schema"""
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 10
                }
            },
            "required": ["query"]
        }

class WebScrapingTool(BaseTool):
    """Tool for scraping web pages"""
    
    def __init__(self):
        super().__init__(
            name="web_scraping",
            description="Scrape content from web pages"
        )
    
    async def execute(self, input_data: WebScrapingInput) -> ToolOutput:
        """Execute web scraping"""
        try:
            timeout = aiohttp.ClientTimeout(total=settings.TOOL_API_TIMEOUT)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                
                async with session.get(input_data.url, headers=headers) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        result = {"url": input_data.url}
                        
                        if input_data.extract_text:
                            # Remove script and style elements
                            for script in soup(["script", "style"]):
                                script.decompose()
                            
                            # Get text content
                            text = soup.get_text()
                            # Clean up whitespace
                            lines = (line.strip() for line in text.splitlines())
                            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                            text = ' '.join(chunk for chunk in chunks if chunk)
                            
                            result["text"] = text[:5000]  # Limit to 5000 chars
                        
                        if input_data.extract_links:
                            links = []
                            for link in soup.find_all('a', href=True):
                                href = link['href']
                                text = link.get_text().strip()
                                if href.startswith('http') or href.startswith('//'):
                                    links.append({"url": href, "text": text})
                                elif href.startswith('/'):
                                    base_url = f"{urlparse(input_data.url).scheme}://{urlparse(input_data.url).netloc}"
                                    links.append({"url": urljoin(base_url, href), "text": text})
                            
                            result["links"] = links[:20]  # Limit to 20 links
                        
                        return ToolOutput(success=True, result=result)
                    else:
                        return ToolOutput(
                            success=False,
                            error=f"Failed to fetch URL. Status: {response.status}"
                        )
        
        except Exception as e:
            logger.error(f"Web scraping error: {str(e)}")
            return ToolOutput(success=False, error=str(e))
    
    def get_parameters_schema(self) -> Dict[str, Any]:
        """Get parameters schema"""
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL to scrape",
                    "format": "uri"
                },
                "extract_text": {
                    "type": "boolean",
                    "description": "Whether to extract text content",
                    "default": True
                },
                "extract_links": {
                    "type": "boolean",
                    "description": "Whether to extract links",
                    "default": False
                }
            },
            "required": ["url"]
        }
