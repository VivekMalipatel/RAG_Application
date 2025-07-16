import io
import asyncio
import csv
import logging
import pandas as pd
from typing import Dict, Any

from core.processors.base_processor import BaseProcessor
from core.markitdown.markdown_handler import MarkDown
from core.model.model_handler import get_global_model_handler
from core.storage.s3_handler import S3Handler

logger = logging.getLogger(__name__)

class StructuredProcessor(BaseProcessor):
    
    def __init__(self):
        self.markdown = MarkDown()
        self.model_handler = get_global_model_handler()
        self.s3_handler = S3Handler()

    async def process(self, task_message) -> Dict[str, Any]:
        raise NotImplementedError("Use process_structured_document method instead")

    async def process_structured_as_text(self, file_data: bytes, s3_base_path: str) -> Dict[str, Any]:
        try:
            loop = asyncio.get_running_loop()
            markdown_text = await loop.run_in_executor(None, self.markdown.convert_bytes, file_data)

            MAX_CHARS = 8000
            batches = []
            
            if len(markdown_text) <= MAX_CHARS:
                batches.append(markdown_text)
            else:
                lines = markdown_text.splitlines(True)
                has_header = False
                try:
                    sample = ''.join(lines[:min(len(lines), 10)])
                    has_header = csv.Sniffer().has_header(sample)
                except Exception:
                    pass
                    
                header = lines[0] if lines and has_header else ''
                data_lines = lines[1:] if len(lines) > 1 and has_header else lines
                current_batch = []
                current_length = len(header)
                
                for line in data_lines:
                    if current_length + len(line) > MAX_CHARS and current_batch:
                        batches.append(header + ''.join(current_batch))
                        current_batch = [line]
                        current_length = len(header) + len(line)
                    else:
                        current_batch.append(line)
                        current_length += len(line)
                        
                if current_batch:
                    batches.append(header + ''.join(current_batch))

            batches_with_entities = []
            for i, batch in enumerate(batches):
                messages = [
                    {
                        "role": "user", 
                        "content": [
                            {"type": "text", "text": batch}
                        ]
                    }
                ]
                
                entities_relationships = await self.model_handler.extract_entities_relationships(messages)
                
                entities, relationships = await self.model_handler.embed_entity_relationship_profiles(
                    entities_relationships["entities"], 
                    entities_relationships["relationships"]
                )
                
                batch_embedding = await self.model_handler.embed_text([batch])
                
                batches_with_entities.append({
                    "page_number": i + 1,
                    "messages": messages,
                    "entities": entities,
                    "relationships": relationships,
                    "image_s3_url": "",
                    "embedding": batch_embedding[0] if batch_embedding else None
                })

            return {
                "data": batches_with_entities,
                "category": "structured"
            }
        except Exception as e:
            logger.error(f"Error processing structured document as text: {e}")
            return {
                "data": [{
                    "page_number": 1,
                    "messages": f"Error processing document: {str(e)}",
                    "entities": [],
                    "relationships": [],
                    "image_s3_url": "",
                    "embedding": None
                }],
                "category": "structured"
            }

    async def process_tabular_sheet(self, df: pd.DataFrame, sheet_name: str, s3_base_path: str) -> Dict[str, Any]:
        try:
            sample_rows = min(20, len(df))
            df_sample = df.head(sample_rows)
            
            dataframe_text = df_sample.to_markdown(index=False)
            
            summary = await self.model_handler.generate_structured_summary(dataframe_text)
            column_profiles = await self.model_handler.generate_column_profiles(dataframe_text)
            
            messages = [
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": f"Summary: {summary}\n\nDataframe:\n{dataframe_text}"}
                    ]
                }
            ]
            
            entities_relationships = await self.model_handler.extract_entities_relationships(messages)
            
            entities, relationships = await self.model_handler.embed_entity_relationship_profiles(
                entities_relationships["entities"], 
                entities_relationships["relationships"]
            )
            
            summary_embedding = await self.model_handler.embed_text([summary])
            column_embeddings = await self.model_handler.embed_text([cp["column_profile"] for cp in column_profiles])
            
            for i, profile in enumerate(column_profiles):
                if i < len(column_embeddings):
                    profile["embedding"] = column_embeddings[i]
            
            return {
                "sheet_name": sheet_name,
                "page_number": 1,
                "messages": messages,
                "entities": entities,
                "relationships": relationships,
                "image_s3_url": "",
                "embedding": summary_embedding[0] if summary_embedding else None,
                "summary": summary,
                "column_profiles": column_profiles,
                "dataframe_sample": dataframe_text,
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "is_tabular": True
            }
        except Exception as e:
            logger.error(f"Error processing tabular sheet {sheet_name}: {e}")
            return {
                "sheet_name": sheet_name,
                "page_number": 1,
                "messages": f"Error processing sheet: {str(e)}",
                "entities": [],
                "relationships": [],
                "image_s3_url": "",
                "embedding": None,
                "summary": f"Error processing sheet: {str(e)}",
                "column_profiles": [],
                "dataframe_sample": "",
                "total_rows": 0,
                "total_columns": 0,
                "is_tabular": False
            }

    async def process_csv_file(self, file_data: bytes, s3_base_path: str) -> Dict[str, Any]:
        try:
            loop = asyncio.get_running_loop()
            
            def read_csv():
                return pd.read_csv(io.BytesIO(file_data))
            
            df = await loop.run_in_executor(None, read_csv)
            
            is_proper_table = self.is_proper_table_structure(df)
            
            if is_proper_table:
                sheet_result = await self.process_tabular_sheet(df, "Sheet1", s3_base_path)
                data = [sheet_result]
            else:
                sheet_result = await self.process_sheet_as_text(df, "Sheet1", s3_base_path)
                if "data" in sheet_result:
                    data = sheet_result["data"]
                else:
                    data = [sheet_result]
            
            return {
                "data": data,
                "category": "structured"
            }
        except Exception as e:
            logger.error(f"Error processing CSV file: {e}")
            return {
                "data": [{
                    "page_number": 1,
                    "messages": f"Error processing CSV file: {str(e)}",
                    "entities": [],
                    "relationships": [],
                    "image_s3_url": "",
                    "embedding": None
                }],
                "category": "structured"
            }

    async def process_sheet_as_text(self, df: pd.DataFrame, sheet_name: str, s3_base_path: str) -> Dict[str, Any]:
        try:
            loop = asyncio.get_running_loop()
            text_content = await loop.run_in_executor(None, lambda: df.to_string())
            
            markdown_text = await loop.run_in_executor(None, self.markdown.convert_text, text_content)
            
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
                
                entities, relationships = await self.model_handler.embed_entity_relationship_profiles(
                    entities_relationships["entities"], 
                    entities_relationships["relationships"]
                )
                
                chunk_embedding = await self.model_handler.embed_text([chunk])
                
                chunks_with_entities.append({
                    "page_number": i + 1,
                    "messages": messages,
                    "entities": entities,
                    "relationships": relationships,
                    "image_s3_url": "",
                    "embedding": chunk_embedding[0] if chunk_embedding else None
                })
            
            return {
                "sheet_name": sheet_name,
                "data": chunks_with_entities,
                "is_tabular": False
            }
        except Exception as e:
            logger.error(f"Error processing sheet as text {sheet_name}: {e}")
            return {
                "sheet_name": sheet_name,
                "data": [{
                    "page_number": 1,
                    "messages": f"Error processing sheet: {str(e)}",
                    "entities": [],
                    "relationships": [],
                    "image_s3_url": "",
                    "embedding": None
                }],
                "is_tabular": False
            }

    def is_proper_table_structure(self, df: pd.DataFrame) -> bool:
        try:
            if df.empty or len(df.columns) < 2:
                return False
            
            non_null_ratio = df.count().sum() / (len(df) * len(df.columns))
            if non_null_ratio < 0.3:
                return False
            
            header_similarity = len(set(df.columns)) / len(df.columns)
            if header_similarity < 0.8:
                return False
            
            return True
        except Exception:
            return False

    async def process_excel_file(self, file_data: bytes, s3_base_path: str) -> Dict[str, Any]:
        try:
            loop = asyncio.get_running_loop()
            
            def read_excel():
                return pd.read_excel(io.BytesIO(file_data), sheet_name=None)
            
            excel_data = await loop.run_in_executor(None, read_excel)
            
            all_sheets_data = []
            for sheet_name, df in excel_data.items():
                is_proper_table = self.is_proper_table_structure(df)
                
                if is_proper_table:
                    sheet_result = await self.process_tabular_sheet(df, sheet_name, s3_base_path)
                    all_sheets_data.append(sheet_result)
                else:
                    sheet_result = await self.process_sheet_as_text(df, sheet_name, s3_base_path)
                    if "data" in sheet_result:
                        all_sheets_data.extend(sheet_result["data"])
                    else:
                        all_sheets_data.append(sheet_result)
            
            return {
                "data": all_sheets_data,
                "category": "structured"
            }
        except Exception as e:
            logger.error(f"Error processing Excel file: {e}")
            return {
                "data": [{
                    "page_number": 1,
                    "messages": f"Error processing Excel file: {str(e)}",
                    "entities": [],
                    "relationships": [],
                    "image_s3_url": "",
                    "embedding": None
                }],
                "category": "structured"
            }

    async def process_structured_document(self, file_data: bytes, file_type: str, s3_base_path: str) -> Dict[str, Any]:
        try:
            if file_type in ['xls', 'xlsx']:
                return await self.process_excel_file(file_data, s3_base_path)
            elif file_type == 'csv':
                return await self.process_csv_file(file_data, s3_base_path)
            else:
                return await self.process_structured_as_text(file_data, s3_base_path)
        except Exception as e:
            logger.error(f"Error processing structured document: {e}")
            return {
                "data": [{
                    "page_number": 1,
                    "messages": f"Error processing structured document: {str(e)}",
                    "entities": [],
                    "relationships": [],
                    "image_s3_url": "",
                    "embedding": None
                }],
                "category": "structured"
            }
