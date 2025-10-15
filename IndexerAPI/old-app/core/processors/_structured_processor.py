import io
import asyncio
import csv
import logging
import pandas as pd
from typing import Dict, Any

from core.processors.base_processor import BaseProcessor
from core.markitdown.markdown_handler import MarkDown
from core.model.model_handler import get_global_model_handler
from core.storage.s3_handler import get_global_s3_handler
from config import settings

logger = logging.getLogger(__name__)

class StructuredProcessor(BaseProcessor):
    
    def __init__(self):
        logger.info("Initializing StructuredProcessor")
        self.markdown = MarkDown()
        self.model_handler = get_global_model_handler()
        logger.info("StructuredProcessor initialized successfully")

    async def process(self, task_message) -> Dict[str, Any]:
        raise NotImplementedError("Use process_structured_document method instead")

    async def process_tabular_sheet(self, df: pd.DataFrame, sheet_name: str, s3_base_path: str) -> Dict[str, Any]:
        logger.info(f"Starting tabular sheet processing for '{sheet_name}' with {len(df)} rows and {len(df.columns)} columns")
        try:
            sample_rows = min(20, len(df))
            df_sample = df.head(sample_rows)
            logger.info(f"Using sample of {sample_rows} rows for processing")
            
            dataframe_text = df_sample.to_markdown(index=False)
            logger.info(f"Generated markdown text with {len(dataframe_text)} characters")
            
            logger.info("Generating summary and column profiles")
            summary_task = self.model_handler.generate_structured_summary(dataframe_text)
            column_profiles_task = self.model_handler.generate_column_profiles(dataframe_text)
            
            summary, column_profiles = await asyncio.gather(
                summary_task,
                column_profiles_task
            )
            logger.info(f"Generated summary {summary[:5000]}... and {len(column_profiles)} column profiles")
            
            messages = [
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": f"Summary: {summary}\n\nDataframe:\n{dataframe_text}"}
                    ]
                }
            ]
            
            logger.info("Generating embeddings for summary and column profiles")
            summary_embedding_task = self.model_handler.embed_text([summary[:settings.EMBEDDING_MAX_TOKENS-1000]])
            column_embeddings_task = self.model_handler.embed_text([cp["column_profile"] for cp in column_profiles])
            
            summary_embedding, column_embeddings = await asyncio.gather(
                summary_embedding_task,
                column_embeddings_task
            )
            logger.debug(f"Generated embeddings: summary={len(summary_embedding)}, columns={len(column_embeddings)}")
            
            for i, profile in enumerate(column_profiles):
                if i < len(column_embeddings):
                    profile["embedding"] = column_embeddings[i]
            
            logger.info(f"Processing {len(df)} rows into row nodes")
            row_nodes = []
            for row_idx, row in df.iterrows():
                row_data = {}
                for col_name in df.columns:
                    cell_value = row[col_name]
                    if pd.notna(cell_value):
                        row_data[col_name] = str(cell_value)
                    else:
                        row_data[col_name] = ""
                
                row_nodes.append({
                    "row_index": int(row_idx),
                    "row_data": row_data
                })
            
            logger.info(f"Successfully processed tabular sheet '{sheet_name}' with {len(row_nodes)} row nodes")
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
                "image_s3_url": "",
                "embedding": None,
                "summary": f"Error processing sheet: {str(e)}",
                "column_profiles": [],
                "row_nodes": [],
                "dataframe_sample": "",
                "total_rows": 0,
                "total_columns": 0,
                "is_tabular": False
            }

    async def process_csv_file(self, file_data: bytes, s3_base_path: str) -> Dict[str, Any]:
        logger.info(f"Starting CSV file processing with {len(file_data)} bytes of data")
        try:
            loop = asyncio.get_running_loop()
            
            def read_csv():
                return pd.read_csv(io.BytesIO(file_data))
            
            logger.debug("Reading CSV data into DataFrame")
            df = await loop.run_in_executor(None, read_csv)
            logger.info(f"Successfully loaded CSV with {len(df)} rows and {len(df.columns)} columns")
            
            is_proper_table = self.is_proper_table_structure(df)
            logger.info(f"CSV structure analysis: is_proper_table={is_proper_table}")
            
            if is_proper_table:
                logger.info("Processing CSV as tabular data")
                sheet_result = await self.process_tabular_sheet(df, "Sheet1", s3_base_path)
                data = [sheet_result]
            else:
                logger.info("Processing CSV as text data")
                sheet_result = await self.process_sheet_as_text(df, "Sheet1", s3_base_path)
                if "data" in sheet_result:
                    data = sheet_result["data"]
                else:
                    data = [sheet_result]
            
            logger.info(f"CSV processing completed successfully with {len(data)} data items")
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
        logger.info(f"Starting text processing for sheet '{sheet_name}' with {len(df)} rows")
        try:
            loop = asyncio.get_running_loop()
            logger.debug("Converting DataFrame to string representation")
            text_content = await loop.run_in_executor(None, lambda: df.to_string())
            logger.debug(f"Generated text content with {len(text_content)} characters")
            
            logger.debug("Converting text to markdown format")
            markdown_text = await loop.run_in_executor(None, self.markdown.convert_text, text_content)
            logger.debug(f"Generated markdown text with {len(markdown_text)} characters")
            
            MAX_CHARS = 8000
            text_chunks = []
            
            if len(markdown_text) <= MAX_CHARS:
                logger.debug("Text fits in single chunk")
                text_chunks.append(markdown_text)
            else:
                logger.info(f"Text exceeds {MAX_CHARS} chars, splitting into chunks")
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
                
                logger.info(f"Split text into {len(text_chunks)} chunks")
            
            logger.info(f"Processing {len(text_chunks)} chunks for entity extraction")
            chunks_with_entities = []
            for i, chunk in enumerate(text_chunks):
                logger.debug(f"Processing chunk {i+1}/{len(text_chunks)} with {len(chunk)} characters")
                messages = [
                    {
                        "role": "user", 
                        "content": [
                            {"type": "text", "text": chunk}
                        ]
                    }
                ]
                
                logger.debug(f"Extracting entities and relationships for chunk {i+1}")
                entities_relationships = await self.model_handler.extract_entities_relationships(messages)
                
                entities, relationships = await self.model_handler.embed_entity_relationship_profiles(
                    entities_relationships["entities"], 
                    entities_relationships["relationships"]
                )
                logger.debug(f"Chunk {i+1}: found {len(entities)} entities and {len(relationships)} relationships")
                
                chunk_embedding = await self.model_handler.embed_text([chunk])
                
                chunks_with_entities.append({
                    "page_number": i + 1,
                    "messages": messages,
                    "entities": entities,
                    "relationships": relationships,
                    "image_s3_url": "",
                    "embedding": chunk_embedding[0] if chunk_embedding else None
                })
            
            logger.info(f"Successfully processed sheet '{sheet_name}' as text with {len(chunks_with_entities)} chunks")
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
        logger.debug(f"Analyzing table structure for DataFrame with shape {df.shape}")
        try:
            if df.empty or len(df.columns) < 2:
                logger.debug("DataFrame is empty or has fewer than 2 columns")
                return False
            
            non_null_ratio = df.count().sum() / (len(df) * len(df.columns))
            logger.debug(f"Non-null ratio: {non_null_ratio:.2f}")
            if non_null_ratio < 0.3:
                logger.debug("Non-null ratio below threshold (0.3)")
                return False
            
            header_similarity = len(set(df.columns)) / len(df.columns)
            logger.debug(f"Header similarity: {header_similarity:.2f}")
            if header_similarity < 0.8:
                logger.debug("Header similarity below threshold (0.8)")
                return False
            
            logger.debug("DataFrame has proper table structure")
            return True
        except Exception as e:
            logger.error(f"Error analyzing table structure: {e}")
            return False

    async def process_excel_file(self, file_data: bytes, s3_base_path: str) -> Dict[str, Any]:
        logger.info(f"Starting Excel file processing with {len(file_data)} bytes of data")
        try:
            loop = asyncio.get_running_loop()
            
            def read_excel():
                return pd.read_excel(io.BytesIO(file_data), sheet_name=None)
            
            logger.debug("Reading Excel data into DataFrames")
            excel_data = await loop.run_in_executor(None, read_excel)
            logger.info(f"Successfully loaded Excel file with {len(excel_data)} sheets: {list(excel_data.keys())}")
            
            all_sheets_data = []
            for sheet_name, df in excel_data.items():
                logger.info(f"Processing sheet '{sheet_name}' with {len(df)} rows and {len(df.columns)} columns")
                is_proper_table = self.is_proper_table_structure(df)
                logger.debug(f"Sheet '{sheet_name}' structure analysis: is_proper_table={is_proper_table}")
                
                if is_proper_table:
                    logger.debug(f"Processing sheet '{sheet_name}' as tabular data")
                    sheet_result = await self.process_tabular_sheet(df, sheet_name, s3_base_path)
                    all_sheets_data.append(sheet_result)
                else:
                    logger.debug(f"Processing sheet '{sheet_name}' as text data")
                    sheet_result = await self.process_sheet_as_text(df, sheet_name, s3_base_path)
                    if "data" in sheet_result:
                        all_sheets_data.extend(sheet_result["data"])
                    else:
                        all_sheets_data.append(sheet_result)
            
            logger.info(f"Excel processing completed successfully with {len(all_sheets_data)} total data items")
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
        logger.info(f"Starting structured document processing for file type '{file_type}' with {len(file_data)} bytes")
        try:
            if file_type in ['xls', 'xlsx']:
                logger.info("Processing as Excel file")
                return await self.process_excel_file(file_data, s3_base_path)
            else:
                logger.info("Processing as CSV file")
                return await self.process_csv_file(file_data, s3_base_path)
            
        except Exception as e:
            logger.error(f"Error processing structured document of type '{file_type}': {e}")
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
