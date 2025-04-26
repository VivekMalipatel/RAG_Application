import logging
import re
from typing import Dict, Any, Optional, Tuple
import asyncio

import requests
from bs4 import BeautifulSoup

from app.core.markitdown import MarkdownConverter
from app.core.model import TextGenerator, EmbeddingGenerator

logger = logging.getLogger(__name__)

class UrlProcessor:
    
    def __init__(self, is_youtube: bool = False, enable_plugins: bool = True):
        self.is_youtube = is_youtube
        self.markdown_converter = MarkdownConverter(enable_plugins=enable_plugins)
        self.text_generator = TextGenerator()
        self.embedding_generator = EmbeddingGenerator()
        logger.info(f"URL processor initialized (YouTube mode: {is_youtube})")
    
    async def process(self, url: str) -> Dict[str, Any]:
        logger.info(f"Processing URL: {url}")
        
        try:
            if not self.is_youtube:
                self.is_youtube = self._is_youtube_url(url)
                
            logger.info(f"URL type detected: {'YouTube' if self.is_youtube else 'Standard web page'}")
            
            if self.is_youtube:
                return await self._process_youtube_url(url)
            else:
                return await self._process_standard_url(url)
                
        except Exception as e:
            logger.error(f"Error processing URL: {str(e)}")
            return {
                "success": False,
                "url": url,
                "error": str(e)
            }
    
    def _is_youtube_url(self, url: str) -> bool:
        youtube_patterns = [
            r"(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=([a-zA-Z0-9_-]+)",
            r"(?:https?:\/\/)?(?:www\.)?youtu\.be\/([a-zA-Z0-9_-]+)"
        ]
        
        for pattern in youtube_patterns:
            if re.search(pattern, url):
                return True
                
        return False
    
    async def _process_youtube_url(self, url: str) -> Dict[str, Any]:
        logger.info(f"Processing YouTube URL: {url}")
        
        try:
            md_result = self.markdown_converter.convert_url(url)
            
            if "error" in md_result.get("metadata", {}):
                logger.error(f"Error extracting YouTube content: {md_result['metadata']['error']}")
                return {
                    "success": False, 
                    "url": url,
                    "error": md_result["metadata"]["error"]
                }
            
            md_content = md_result["text_content"]
            metadata = md_result["metadata"]
            
            enriched_content = await self._analyze_with_vlm(md_content, url)
            
            embedding_result = await self._generate_embeddings(md_content)
            
            summary = await self._generate_summary(md_content, is_youtube=True)
            
            return {
                "url": url,
                "is_youtube": True,
                "markdown_content": md_content,
                "html_content": "",
                "enriched_content": enriched_content,
                "summary": summary,
                "metadata": metadata,
                "embedding_id": "embedding_placeholder_id",
                "token_count": embedding_result.get("usage", {}).get("total_tokens", 0),
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error processing YouTube URL: {str(e)}")
            return {
                "success": False,
                "url": url,
                "error": str(e)
            }
    
    async def _process_standard_url(self, url: str) -> Dict[str, Any]:
        logger.info(f"Processing standard URL: {url}")
        
        try:
            html_content, page_title = await self._browse_page(url)
            
            md_result = self.markdown_converter.convert_url(url)
            
            if "error" in md_result.get("metadata", {}):
                logger.error(f"Error converting URL to markdown: {md_result['metadata']['error']}")
                md_content = self._html_to_markdown(html_content, page_title)
            else:
                md_content = md_result["text_content"]
            
            enriched_content = await self._analyze_with_vlm(md_content, url)
            
            embedding_result = await self._generate_embeddings(md_content)
            
            summary = await self._generate_summary(md_content, is_youtube=False)
            
            return {
                "url": url,
                "is_youtube": False,
                "markdown_content": md_content,
                "html_content": html_content[:10000] if len(html_content) > 10000 else html_content,
                "enriched_content": enriched_content,
                "summary": summary,
                "metadata": {
                    "page_title": page_title,
                    "content_length": len(html_content)
                },
                "embedding_id": "embedding_placeholder_id",
                "token_count": embedding_result.get("usage", {}).get("total_tokens", 0),
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error processing standard URL: {str(e)}")
            return {
                "success": False,
                "url": url,
                "error": str(e)
            }
    
    async def _browse_page(self, url: str) -> Tuple[str, str]:
        logger.info(f"Browsing web page: {url}")
        
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, "html.parser")
            
            page_title = soup.title.string if soup.title else "Untitled Page"
            
            for script_or_style in soup(["script", "style"]):
                script_or_style.extract()
                
            html_content = str(soup)
            
            logger.info(f"Successfully extracted HTML content (length: {len(html_content)}) and title: {page_title}")
            
            return html_content, page_title
            
        except Exception as e:
            logger.error(f"Error browsing page: {str(e)}")
            raise
    
    def _html_to_markdown(self, html_content: str, page_title: str) -> str:
        logger.info("Converting HTML to markdown using fallback method")
        
        try:
            soup = BeautifulSoup(html_content, "html.parser")
            
            markdown = f"# {page_title}\n\n"
            
            paragraphs = soup.find_all("p")
            for p in paragraphs:
                markdown += f"{p.get_text()}\n\n"
            
            for heading_level in range(1, 7):
                headings = soup.find_all(f"h{heading_level}")
                for h in headings:
                    markdown += f"{'#' * heading_level} {h.get_text()}\n\n"
            
            for list_tag in soup.find_all(["ul", "ol"]):
                is_ordered = list_tag.name == "ol"
                for item in list_tag.find_all("li"):
                    prefix = "1. " if is_ordered else "- "
                    markdown += f"{prefix}{item.get_text()}\n"
                markdown += "\n"
            
            return markdown
            
        except Exception as e:
            logger.error(f"Error in fallback HTML to markdown conversion: {str(e)}")
            return soup.get_text()
    
    async def _analyze_with_vlm(self, content: str, url: str) -> str:
        logger.info(f"Analyzing content with VLM for URL: {url}")
        
        try:
            prompt = f"""
            Analyze the following content from URL: {url}
            Extract key information, identify main topics, and provide insights.
            
            Content:
            {content[:4000]}
            """
            
            system_message = "You are a helpful assistant that analyzes web content."
            
            result = await self.text_generator.generate_text(
                prompt=prompt,
                system_message=system_message,
                temperature=0.3
            )
            
            if result["success"]:
                return result["content"]
            else:
                logger.error(f"Error analyzing with VLM: {result.get('error', 'Unknown error')}")
                return "Content analysis failed."
                
        except Exception as e:
            logger.error(f"Error analyzing with VLM: {str(e)}")
            return "Content analysis failed."
    
    async def _generate_embeddings(self, text: str) -> Dict[str, Any]:
        try:
            logger.info("Generating embeddings for URL content")
            
            text_for_embedding = text[:8000]
            
            embedding_result = await self.embedding_generator.generate_embedding(text_for_embedding)
            
            return embedding_result
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _generate_summary(self, text: str, is_youtube: bool = False) -> str:
        try:
            logger.info(f"Generating summary for {'YouTube' if is_youtube else 'web'} content")
            
            content_type = "YouTube video" if is_youtube else "web page"
            
            prompt = f"""
            Please provide a concise summary (2-3 sentences) of the following {content_type} content:
            
            {text[:4000]}
            """
            
            system_message = "You are a helpful assistant that generates concise summaries."
            
            result = await self.text_generator.generate_text(
                prompt=prompt,
                system_message=system_message,
                temperature=0.3,
                max_tokens=200
            )
            
            if result["success"]:
                return result["content"]
            else:
                logger.error(f"Error generating summary: {result.get('error', 'Unknown error')}")
                return "Summary generation failed."
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return "Summary generation failed."