import asyncio
import logging
from typing import Any, Dict
from core.processors.base_processor import BaseProcessor
from core.model.model_handler import get_global_model_handler
from core.storage.neo4j_handler import get_neo4j_handler

logger = logging.getLogger(__name__)

class DirectProcessor(BaseProcessor):
    def __init__(self):
        self.model_handler = get_global_model_handler()
        self.neo4j_handler = get_neo4j_handler()

    async def process(self, task_message) -> Dict[str, Any]:
        try:
            payload = task_message.payload
            document = payload["document"]
            chunk_index = payload["chunk_index"]
            chunk_text = payload["text"]
            messages = [{"role": "user", "content": [{"type": "text", "text": chunk_text}]}]
            entities_relationships = await self.model_handler.extract_entities_relationships(messages)
            entities_task = self.model_handler.embed_entity_relationship_profiles(
                entities_relationships["entities"],
                entities_relationships["relationships"],
            )
            embedding_task = self.model_handler.embed_text([chunk_text])
            (entities, relationships), embedding_result = await asyncio.gather(
                entities_task,
                embedding_task,
            )
            page_data = {
                "page_number": chunk_index,
                "messages": messages,
                "entities": entities,
                "relationships": relationships,
                "image_s3_url": "",
                "embedding": embedding_result[0] if embedding_result else None,
                "processing_task_id": task_message.task_id,
            }
            await self.neo4j_handler.upsert_direct_chunk(document, page_data)
            return page_data
        except Exception as exc:
            logger.error(f"Error processing direct chunk task {task_message.task_id}: {exc}")
            raise
