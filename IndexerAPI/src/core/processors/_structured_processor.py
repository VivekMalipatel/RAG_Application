import io
import asyncio
import logging
from typing import Any, Dict, List
import pandas as pd
from core.processors.base_processor import BaseProcessor
from core.markitdown.markdown_handler import MarkDown
from core.model.model_handler import get_global_model_handler
from core.storage.s3_handler import get_global_s3_handler
from core.storage.neo4j_handler import get_neo4j_handler

logger = logging.getLogger(__name__)

class StructuredProcessor(BaseProcessor):
    def __init__(self):
        logger.info("Initializing StructuredProcessor")
        self.markdown = MarkDown()
        self.model_handler = get_global_model_handler()
        self.s3_handler = None
        self.neo4j_handler = get_neo4j_handler()
        logger.info("StructuredProcessor initialized successfully")

    async def process(self, task_message) -> Dict[str, Any]:
        payload = task_message.payload
        document = payload["document"]
        sheet_name = payload["sheet_name"]
        sheet_key = payload["sheet_s3_key"]
        if not self.s3_handler:
            self.s3_handler = await get_global_s3_handler()
        sheet_bytes = await self.s3_handler.download_bytes(sheet_key)
        dataframe = pd.read_csv(io.BytesIO(sheet_bytes))
        is_tabular = self.is_proper_table_structure(dataframe)
        if is_tabular:
            sheet_result = await self.process_tabular_sheet(dataframe, sheet_name)
            sheet_result["processing_task_id"] = task_message.task_id
            await self.neo4j_handler.upsert_structured_sheet(document, sheet_result)
            return sheet_result
        text_result = await self.process_sheet_as_text(dataframe, sheet_name)
        processed_chunks = []
        for chunk in text_result.get("data", []):
            chunk["processing_task_id"] = task_message.task_id
            await self.neo4j_handler.upsert_structured_text_chunk(document, sheet_name, chunk)
            processed_chunks.append(chunk)
        return {"sheet_name": sheet_name, "data": processed_chunks, "is_tabular": False}

    async def process_tabular_sheet(self, dataframe: pd.DataFrame, sheet_name: str) -> Dict[str, Any]:
        logger.info(
            f"Starting tabular sheet processing for '{sheet_name}' with {len(dataframe)} rows and {len(dataframe.columns)} columns"
        )
        try:
            sample_rows = min(20, len(dataframe))
            dataframe_sample = dataframe.head(sample_rows)
            logger.info(f"Using sample of {sample_rows} rows for processing")
            dataframe_text = dataframe_sample.to_markdown(index=False)
            logger.info(f"Generated markdown text with {len(dataframe_text)} characters")
            summary_task = self.model_handler.generate_structured_summary(dataframe_text)
            column_profiles_task = self.model_handler.generate_column_profiles(dataframe_text)
            summary, column_profiles = await asyncio.gather(summary_task, column_profiles_task)
            logger.info(f"Generated summary {summary[:5000]}... and {len(column_profiles)} column profiles")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Summary: {summary}\n\nDataframe:\n{dataframe_text}"}
                    ],
                }
            ]
            summary_embedding_task = self.model_handler.embed_text([summary])
            column_embeddings_task = self.model_handler.embed_text(
                [profile["column_profile"] for profile in column_profiles]
            )
            summary_embedding, column_embeddings = await asyncio.gather(
                summary_embedding_task,
                column_embeddings_task,
            )
            for index, profile in enumerate(column_profiles):
                if index < len(column_embeddings):
                    profile["embedding"] = column_embeddings[index]
            row_nodes = []
            for row_idx, row in dataframe.iterrows():
                row_data: Dict[str, str] = {}
                for column_name in dataframe.columns:
                    cell_value = row[column_name]
                    if pd.notna(cell_value):
                        row_data[column_name] = str(cell_value)
                    else:
                        row_data[column_name] = ""
                row_nodes.append({"row_index": int(row_idx), "row_data": row_data})
            logger.info(
                f"Successfully processed tabular sheet '{sheet_name}' with {len(row_nodes)} row nodes"
            )
            return {
                "sheet_name": sheet_name,
                "page_number": 1,
                "messages": messages,
                "image_s3_url": "",
                "embedding": summary_embedding[0] if summary_embedding else None,
                "summary": summary,
                "column_profiles": column_profiles,
                "row_nodes": row_nodes,
                "dataframe_sample": dataframe_text,
                "total_rows": len(dataframe),
                "total_columns": len(dataframe.columns),
                "is_tabular": True,
            }
        except Exception as exc:
            logger.error(f"Error processing tabular sheet {sheet_name}: {exc}")
            raise

    async def process_sheet_as_text(self, dataframe: pd.DataFrame, sheet_name: str) -> Dict[str, Any]:
        logger.info(f"Starting text processing for sheet '{sheet_name}' with {len(dataframe)} rows")
        try:
            loop = asyncio.get_running_loop()
            text_content = await loop.run_in_executor(None, lambda: dataframe.to_string())
            markdown_text = await loop.run_in_executor(None, self.markdown.convert_text, text_content)
            max_chars = 8000
            text_chunks: List[str] = []
            if len(markdown_text) <= max_chars:
                text_chunks.append(markdown_text)
            else:
                words = markdown_text.split()
                current_chunk: List[str] = []
                current_length = 0
                for word in words:
                    if current_length + len(word) + 1 > max_chars and current_chunk:
                        text_chunks.append(" ".join(current_chunk))
                        current_chunk = [word]
                        current_length = len(word)
                    else:
                        current_chunk.append(word)
                        current_length += len(word) + 1
                if current_chunk:
                    text_chunks.append(" ".join(current_chunk))
            chunks_with_entities = []
            for index, chunk in enumerate(text_chunks):
                messages = [{"role": "user", "content": [{"type": "text", "text": chunk}]}]
                entities_relationships = await self.model_handler.extract_entities_relationships(messages)
                entities, relationships = await self.model_handler.embed_entity_relationship_profiles(
                    entities_relationships["entities"],
                    entities_relationships["relationships"],
                )
                chunk_embedding = await self.model_handler.embed_text([chunk])
                chunks_with_entities.append(
                    {
                        "page_number": index + 1,
                        "messages": messages,
                        "entities": entities,
                        "relationships": relationships,
                        "image_s3_url": "",
                        "embedding": chunk_embedding[0] if chunk_embedding else None,
                    }
                )
            return {"sheet_name": sheet_name, "data": chunks_with_entities, "is_tabular": False}
        except Exception as exc:
            logger.error(f"Error processing sheet as text {sheet_name}: {exc}")
            raise

    def is_proper_table_structure(self, dataframe: pd.DataFrame) -> bool:
        logger.debug(f"Analyzing table structure for DataFrame with shape {dataframe.shape}")
        try:
            if dataframe.empty or len(dataframe.columns) < 2:
                return False
            non_null_ratio = dataframe.count().sum() / (len(dataframe) * len(dataframe.columns))
            if non_null_ratio < 0.3:
                return False
            header_similarity = len(set(dataframe.columns)) / len(dataframe.columns)
            if header_similarity < 0.8:
                return False
            return True
        except Exception as exc:
            logger.error(f"Error analyzing table structure: {exc}")
            return False
