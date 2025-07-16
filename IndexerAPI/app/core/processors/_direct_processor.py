import asyncio
import logging
from typing import Dict, Any

from core.processors.base_processor import BaseProcessor
from core.markitdown.markdown_handler import MarkDown
from core.model.model_handler import get_global_model_handler
from core.storage.s3_handler import get_global_s3_handler

logger = logging.getLogger(__name__)

class DirectProcessor(BaseProcessor):
    
    def __init__(self):
        self.markdown = MarkDown()
        self.model_handler = get_global_model_handler()

    async def process(self, task_message) -> Dict[str, Any]:
        raise NotImplementedError("Use process_direct_document method instead")

    async def process_direct_document(self, file_data: bytes, file_type: str, s3_base_path: str) -> Dict[str, Any]:
        try:
            loop = asyncio.get_running_loop()
            text = await loop.run_in_executor(None, lambda: file_data.decode('utf-8', errors='replace'))
            
            if file_type == 'markdown':
                markdown_text = text
            else:
                markdown_text = await loop.run_in_executor(None, self.markdown.convert_text, text)
            
            MAX_CHARS = 8000
            text_chunks = []
            
            if len(markdown_text) <= MAX_CHARS:
                text_chunks.append(markdown_text)
            else:
                words = markdown_text.split()
                current_chunk = []
                current_length = 0
                
                for word in words:
                    if current_length + len(word) + 1 > MAX_CHARS and current_chunk:
                        text_chunks.append(' '.join(current_chunk))
                        current_chunk = [word]
                        current_length = len(word)
                    else:
                        current_chunk.append(word)
                        current_length += len(word) + 1
                
                if current_chunk:
                    text_chunks.append(' '.join(current_chunk))
            
            chunks_with_entities = []
            for i, chunk in enumerate(text_chunks):
                messages = [
                    {
                        "role": "user", 
                        "content": [
                            {"type": "text", "text": chunk}
                        ]
                    }
                ]
                
                entities_relationships = await self.model_handler.extract_entities_relationships(messages)
                
                entities_relationships_task = self.model_handler.embed_entity_relationship_profiles(
                    entities_relationships["entities"], 
                    entities_relationships["relationships"]
                )
                chunk_embedding_task = self.model_handler.embed_text([chunk])
                
                (entities, relationships), chunk_embedding = await asyncio.gather(
                    entities_relationships_task,
                    chunk_embedding_task
                )
                
                chunks_with_entities.append({
                    "page_number": i + 1,
                    "messages": messages,
                    "entities": entities,
                    "relationships": relationships,
                    "image_s3_url": "",
                    "embedding": chunk_embedding[0] if chunk_embedding else None
                })
            
            return {
                "data": chunks_with_entities,
                "category": "direct"
            }
        except Exception as e:
            logger.error(f"Error processing direct document: {e}")
            return {
                "data": [{
                    "page_number": 1,
                    "messages": f"Error processing document: {str(e)}",
                    "entities": [],
                    "relationships": [],
                    "image_s3_url": "",
                    "embedding": None
                }],
                "category": "direct"
            }
