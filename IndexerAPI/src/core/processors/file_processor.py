import asyncio
import io
import logging
import os
import sys
import uuid
import pandas as pd
import pypdf
from typing import Dict, List
from core.processors.base_processor import BaseProcessor
from core.processors.utils import detect_file_type, download_file_from_s3_url, convert_to_pdf
from core.markitdown.markdown_handler import MarkDown
from core.storage.neo4j_handler import get_neo4j_handler
from core.storage.s3_handler import get_global_s3_handler
from core.queue.task_types import TaskMessage, TaskType
from core.queue.rabbitmq_handler import rabbitmq_handler
from core.config import settings

logger = logging.getLogger(__name__)

_REQUIRED_RECURSION_LIMIT = 10000
if sys.getrecursionlimit() < _REQUIRED_RECURSION_LIMIT:
    sys.setrecursionlimit(_REQUIRED_RECURSION_LIMIT)

def _extract_page_bytes(reader: pypdf.PdfReader, index: int) -> bytes:
    writer = pypdf.PdfWriter()
    writer.add_page(reader.pages[index])
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()

class FileProcessor(BaseProcessor):
    def __init__(self):
        logger.info("Initializing FileProcessor")
        self.markdown = MarkDown()
        self.neo4j_handler = get_neo4j_handler()
        self.unoserver_host = settings.UNOSERVER_HOST
        self.unoserver_port = settings.UNOSERVER_PORT
        self.unstructured_docs = [
            "pdf",
            "doc",
            "docx",
            "ppt",
            "txtascii",
            "txtutf8",
            "txtutf16",
            "pptx",
            "jpeg",
            "jpg",
            "png",
        ]
        self.structured_docs = ["csv", "xls", "xlsx"]
        self.direct_processing_docs = [
            "txt",
            "markdown",
            "json",
            "python",
            "java",
            "go",
            "ruby",
            "php",
            "bash",
            "shell",
            "c",
            "javascript",
            "cpp",
            "html",
            "css",
            "xml",
            "yaml",
            "toml",
        ]
        logger.info(
            f"FileProcessor initialized with {len(self.unstructured_docs)} unstructured, "
            f"{len(self.structured_docs)} structured, and {len(self.direct_processing_docs)} direct processing file types"
        )

    def _safe_filename(self, filename: str) -> str:
        return "".join(c if c.isalnum() or c in ["-", "_"] else "_" for c in filename)

    def _build_document_context(self, task_message: TaskMessage, payload: dict, file_type: str, category: str, file_size: int) -> dict:
        user_id = payload.get("user_id")
        org_id = payload.get("org_id")
        source = payload.get("source")
        metadata = payload.get("metadata", {})
        filename = metadata.get("filename", f"file_{uuid.uuid1()}")
        internal_object_id = f"{org_id}_{user_id}_{source}_{filename}"
        base_filename = os.path.splitext(filename)[0]
        s3_base_path = f"{org_id}/{user_id}/{source}/{self._safe_filename(base_filename)}"
        document = {
            "internal_object_id": internal_object_id,
            "user_id": user_id,
            "org_id": org_id,
            "source": source,
            "filename": filename,
            "file_type": file_type,
            "category": category,
            "s3_url": payload.get("s3_url"),
            "metadata": metadata,
            "base_task_id": task_message.task_id,
            "s3_base_path": s3_base_path,
        }
        doc_properties = {
            "internal_object_id": internal_object_id,
            "user_id": user_id,
            "org_id": org_id,
            "source": source,
            "filename": filename,
            "file_type": file_type,
            "category": category,
            "s3_url": payload.get("s3_url"),
            "task_id": task_message.task_id,
            "file_size_bytes": file_size,
        }
        for key, value in metadata.items():
            if key not in doc_properties:
                doc_properties[f"metadata_{key}"] = value
        return {"document": document, "document_properties": doc_properties, "s3_base_path": s3_base_path}

    def categorize_file(self, file_type: str) -> str:
        logger.debug(f"Categorizing file type: {file_type}")
        if file_type in self.unstructured_docs:
            logger.debug(f"File type '{file_type}' categorized as unstructured")
            return "unstructured"
        if file_type in self.structured_docs:
            logger.debug(f"File type '{file_type}' categorized as structured")
            return "structured"
        if file_type in self.direct_processing_docs:
            logger.debug(f"File type '{file_type}' categorized as direct")
            return "direct"
        logger.error(f"Unsupported file type encountered: {file_type}")
        raise ValueError(f"Unsupported File type: {file_type}")

    async def process(self, task_message: TaskMessage) -> None:
        logger.info(f"Starting file processing for task {task_message.task_id}")
        payload = task_message.payload
        s3_url = payload.get("s3_url")
        if not s3_url:
            raise ValueError("No S3 URL provided in task message")
        file_data = await download_file_from_s3_url(s3_url)
        file_type = await detect_file_type(file_data)
        category = self.categorize_file(file_type)
        context = self._build_document_context(task_message, payload, file_type, category, len(file_data))
        await self.neo4j_handler.reset_document(context["document_properties"])
        if category == "unstructured":
            await self._fan_out_unstructured(context, file_data, file_type)
        elif category == "structured":
            await self._fan_out_structured(context, file_data, file_type)
        else:
            await self._fan_out_direct(context, file_data)

    async def _fan_out_unstructured(self, context: dict, file_data: bytes, file_type: str) -> None:
        pdf_bytes = await convert_to_pdf(file_data, file_type)
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        total_pages = len(reader.pages)
        s3_handler = await get_global_s3_handler()
        s3_base_path = context["s3_base_path"]
        document = context["document"]
        fanout_limit = settings.UNSTRUCTURED_FANOUT_CONCURRENCY or settings.MAX_DEQUEUE_CONCURRENCY
        semaphore = asyncio.Semaphore(max(1, fanout_limit))

        async def process_page(index: int):
            async with semaphore:
                try:
                    page_bytes = await asyncio.to_thread(_extract_page_bytes, reader, index)
                except RecursionError as exc:
                    raise RuntimeError(f"Failed to split page {index + 1}: recursion depth exceeded") from exc

                page_key = f"{s3_base_path}/pages/page_{index + 1}.pdf"
                await s3_handler.upload_bytes(page_bytes, page_key)
                chunk_payload = {
                    "document": document,
                    "page_number": index + 1,
                    "page_s3_key": page_key,
                    "total_pages": total_pages,
                    "s3_base_path": s3_base_path,
                }
                chunk_task = TaskMessage(task_id=str(uuid.uuid4()), task_type=TaskType.UNSTRUCTURED_PAGE, payload=chunk_payload)
                await rabbitmq_handler.enqueue_task(chunk_task)

        async with asyncio.TaskGroup() as task_group:
            for page_index in range(total_pages):
                task_group.create_task(process_page(page_index))
        logger.info(
            f"Queued {total_pages} unstructured pages for {document['internal_object_id']}"
        )

    async def _fan_out_structured(self, context: dict, file_data: bytes, file_type: str) -> None:
        s3_handler = await get_global_s3_handler()
        document = context["document"]
        s3_base_path = context["s3_base_path"]
        sheets: Dict[str, pd.DataFrame]
        if file_type in ["xls", "xlsx"]:
            sheets = pd.read_excel(io.BytesIO(file_data), sheet_name=None)
        else:
            dataframe = pd.read_csv(io.BytesIO(file_data))
            sheets = {"Sheet1": dataframe}

        async def process_sheet(name: str, dataframe: pd.DataFrame):
            sheet_key = f"{s3_base_path}/structured/{self._safe_filename(name)}.csv"
            csv_bytes = dataframe.to_csv(index=False).encode("utf-8")
            await s3_handler.upload_bytes(csv_bytes, sheet_key)
            chunk_payload = {
                "document": document,
                "sheet_name": name,
                "sheet_s3_key": sheet_key,
                "s3_base_path": s3_base_path,
            }
            chunk_task = TaskMessage(task_id=str(uuid.uuid4()), task_type=TaskType.STRUCTURED_CHUNK, payload=chunk_payload)
            await rabbitmq_handler.enqueue_task(chunk_task)

        await asyncio.gather(*[process_sheet(name, df) for name, df in sheets.items()])

    async def _fan_out_direct(self, context: dict, file_data: bytes) -> None:
        document = context["document"]
        try:
            text = file_data.decode("utf-8", errors="replace")
        except Exception:
            text = file_data.decode("latin-1", errors="replace")
        if document["file_type"] == "markdown":
            markdown_text = text
        else:
            markdown_text = self.markdown.convert_text(text)
        max_chars = 8000
        chunks: List[str] = []
        if len(markdown_text) <= max_chars:
            chunks.append(markdown_text)
        else:
            words = markdown_text.split()
            current_chunk: List[str] = []
            current_length = 0
            for word in words:
                word_len = len(word)
                if current_length + word_len + 1 > max_chars and current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [word]
                    current_length = word_len
                else:
                    current_chunk.append(word)
                    current_length += word_len + 1
            if current_chunk:
                chunks.append(" ".join(current_chunk))

        async def process_chunk(index: int, chunk_text: str):
            chunk_payload = {
                "document": document,
                "chunk_index": index + 1,
                "text": chunk_text,
            }
            chunk_task = TaskMessage(task_id=str(uuid.uuid4()), task_type=TaskType.DIRECT_CHUNK, payload=chunk_payload)
            await rabbitmq_handler.enqueue_task(chunk_task)

        await asyncio.gather(*[process_chunk(idx, chunk) for idx, chunk in enumerate(chunks)])
