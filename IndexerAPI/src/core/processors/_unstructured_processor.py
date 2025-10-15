import asyncio
import base64
import logging
from typing import Any, Dict
from core.processors.base_processor import BaseProcessor
from core.processors.utils import rasterize_and_encode
from core.markitdown.markdown_handler import MarkDown
from core.model.model_handler import get_global_model_handler
from core.storage.s3_handler import get_global_s3_handler
from core.storage.neo4j_handler import get_neo4j_handler

logger = logging.getLogger(__name__)

class UnstructuredProcessor(BaseProcessor):
    def __init__(self):
        logger.info("Initializing UnstructuredProcessor")
        self.markdown = MarkDown()
        self.model_handler = get_global_model_handler()
        self.neo4j_handler = get_neo4j_handler()
        logger.info("UnstructuredProcessor initialized successfully")

    async def process(self, task_message) -> Dict[str, Any]:
        try:
            payload = task_message.payload
            document = payload["document"]
            page_number = payload["page_number"]
            page_key = payload["page_s3_key"]
            s3_base_path = payload.get("s3_base_path", document.get("s3_base_path", ""))
            s3_handler = await get_global_s3_handler()
            page_bytes = await s3_handler.download_bytes(page_key)
            page_data = await self._process_page(page_bytes, page_number, s3_base_path, s3_handler)
            page_data["processing_task_id"] = task_message.task_id
            await self.neo4j_handler.upsert_unstructured_page(document, page_data)
            return page_data
        except Exception as exc:
            logger.error(f"Error processing unstructured page task {task_message.task_id}: {exc}")
            raise

    async def _process_page(self, page_bytes: bytes, page_number: int, s3_base_path: str, s3_handler) -> Dict[str, Any]:
        loop = asyncio.get_running_loop()
        extracted_text_task = loop.run_in_executor(None, self.markdown.convert_bytes, page_bytes)
        image_task = loop.run_in_executor(None, rasterize_and_encode, page_bytes, page_number - 1)
        extracted_text, (_, image_base64) = await asyncio.gather(extracted_text_task, image_task)
        text_description = await self.model_handler.generate_text_description(image_base64)
        combined_text = f"description: {text_description}\n\nExtracted text from page: {extracted_text}"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                    {"type": "text", "text": combined_text},
                ],
            }
        ]
        entities_task = self.model_handler.extract_entities_relationships(messages)
        embed_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                    {"type": "text", "text": "truncated extracted text and summary:" + combined_text[:1000]},
                ],
            }
        ]
        embedding_task = self.model_handler.embed(embed_messages)
        entities_relationships, page_embedding = await asyncio.gather(entities_task, embedding_task)
        entities, relationships = await self.model_handler.embed_entity_relationship_profiles(
            entities_relationships["entities"],
            entities_relationships["relationships"],
        )
        image_bytes = base64.b64decode(image_base64)
        image_key = f"metadata/{s3_base_path}/page_{page_number}.jpg"
        await s3_handler.upload_bytes(image_bytes, image_key)
        image_url = f"{s3_handler.endpoint_url}/{s3_handler.bucket_name}/{image_key}"
        return {
            "page_number": page_number,
            "messages": messages,
            "entities": entities,
            "relationships": relationships,
            "image_s3_url": image_url,
            "embedding": page_embedding[0] if page_embedding else None,
        }
