import logging
from typing import Dict, Any, Optional
import asyncio

from app.core.markitdown import MarkdownConverter
from app.core.model import TextGenerator, EmbeddingGenerator

logger = logging.getLogger(__name__)

class TextProcessor:
    
    def __init__(self, enable_plugins: bool = False):
        self.markdown_converter = MarkdownConverter(enable_plugins=enable_plugins)
        self.text_generator = TextGenerator()
        self.embedding_generator = EmbeddingGenerator()
        logger.info("Text processor initialized")
        
    async def process(self, text: str) -> Dict[str, Any]:
        logger.info(f"Processing raw text (length: {len(text)})")
        
        try:
            md_content = await self._convert_to_markdown(text)
            
            enriched_content = await self._enrich_with_llm(md_content)
            
            embeddings_result = await self._generate_embeddings(md_content)
            
            summary = await self._generate_summary(md_content)
            
            return {
                "original_text": text[:1000] + "..." if len(text) > 1000 else text,
                "markdown_content": md_content,
                "enriched_content": enriched_content.get("content", ""),
                "summary": summary,
                "token_count": embeddings_result.get("usage", {}).get("total_tokens", 0),
                "embedding_id": "embedding_placeholder_id",
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error processing text: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _convert_to_markdown(self, text: str) -> str:
        has_md_indicators = any(indicator in text for indicator in ["#", "```", "*", "_", ">", "-", "1.", "[", "!["])
        
        if has_md_indicators:
            logger.info("Text appears to already be in markdown format")
            return text
        
        try:
            logger.info("Converting raw text to markdown")
            lines = text.split("\n")
            md_lines = []
            
            in_paragraph = False
            
            for line in lines:
                line = line.strip()
                
                if not line:
                    if in_paragraph:
                        md_lines.append("")
                        in_paragraph = False
                    continue
                
                if line.isupper() and len(line) < 100:
                    md_lines.append(f"## {line}")
                elif len(line) < 50 and line.endswith(":"):
                    md_lines.append(f"### {line}")
                else:
                    md_lines.append(line)
                    in_paragraph = True
            
            return "\n".join(md_lines)
            
        except Exception as e:
            logger.error(f"Error converting to markdown: {str(e)}")
            return text
    
    async def _enrich_with_llm(self, markdown_text: str) -> Dict[str, Any]:
        try:
            logger.info("Enriching markdown content with LLM")
            
            prompt = f"""
            Analyze the following markdown content and enhance it by:
            1. Identifying key entities and concepts
            2. Adding appropriate headers if missing
            3. Organizing content for better readability
            
            Here's the content to analyze:
            
            {markdown_text[:4000]}
            """
            
            system_message = "You are a helpful assistant that analyzes and improves markdown content."
            
            result = await self.text_generator.generate_text(
                prompt=prompt,
                system_message=system_message,
                temperature=0.3
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error enriching with LLM: {str(e)}")
            return {"success": False, "content": markdown_text, "error": str(e)}
    
    async def _generate_embeddings(self, text: str) -> Dict[str, Any]:
        try:
            logger.info("Generating embeddings for processed text")
            
            text_for_embedding = text[:8000]
            
            embedding_result = await self.embedding_generator.generate_embedding(text_for_embedding)
            
            return embedding_result
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _generate_summary(self, text: str) -> str:
        try:
            logger.info("Generating summary for text")
            
            prompt = f"""
            Please provide a concise summary (2-3 sentences) of the following content:
            
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