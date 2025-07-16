import os
import logging
import uuid
from concurrent.futures import ProcessPoolExecutor

from core.processors.base_processor import BaseProcessor
from core.processors.utils import detect_file_type, download_file_from_s3_url
from core.processors._direct_processor import DirectProcessor
from core.processors._structured_processor import StructuredProcessor
from core.processors._unstructured_processor import UnstructuredProcessor
from core.markitdown.markdown_handler import MarkDown
from core.model.model_handler import get_global_model_handler
from core.storage.s3_handler import get_global_s3_handler
from core.storage.neo4j_handler import get_neo4j_handler
from core.queue.task_types import TaskMessage
from config import settings

logger = logging.getLogger(__name__)

class FileProcessor(BaseProcessor):
    
    def __init__(self):
        self.markdown = MarkDown()
        self.model_handler = get_global_model_handler()
        self.neo4j_handler = get_neo4j_handler()
        
        self.direct_processor = DirectProcessor()
        self.structured_processor = StructuredProcessor()
        self.unstructured_processor = UnstructuredProcessor()
        
        self.unoserver_host = settings.UNOSERVER_HOST
        self.unoserver_port = settings.UNOSERVER_PORT
        
        self.unstructured_docs = [
            'pdf', 
            'doc',
            'docx',
            'ppt',
            'txtascii',
            'txtutf8',
            'txtutf16',
            'pptx',
            'jpeg',
            'jpg',
            'png'
        ]
        
        self.structured_docs = [
            'csv',
            'xls',
            'xlsx'
        ]

        self.direct_processing_docs = [
            'txt',
            'markdown',
            'json',
            'python',
            'java',
            'go',
            'ruby',
            'php',
            'bash',
            'shell',
            'c',
            'javascript',
            'cpp',
            'html',
            'css',
            'xml',
            'yaml',
            'toml',
        ]

    def categorize_file(self, file_type: str) -> str:
        if file_type in self.unstructured_docs:
            return 'unstructured'
        elif file_type in self.structured_docs:
            return 'structured'
        elif file_type in self.direct_processing_docs:
            return 'direct'
        else:
            raise ValueError(f"Unsupported File type: {file_type}")

    async def process(self, task_message: TaskMessage) -> None:
        try:
            payload = task_message.payload
            s3_url = payload.get('s3_url')
            
            if not s3_url:
                raise ValueError("No S3 URL provided in task message")
            
            logger.info(f"Processing file from S3 URL: {s3_url}")
            
            file_data = await download_file_from_s3_url(s3_url)
            
            file_type = await detect_file_type(file_data)
            category = self.categorize_file(file_type)
            
            logger.info(f"File type: {file_type}, Category: {category}")
            
            source = payload.get('source')
            user_id = payload.get('user_id')
            org_id = payload.get('org_id')
            
            source_filename = payload.get('metadata', {}).get('filename', f'file_{str(uuid.uuid1())}')
            internal_object_id = f"{org_id}_{user_id}_{source}_{source_filename}"
            
            base_filename = os.path.splitext(source_filename)[0]
            s3_friendly_folder_name = "".join(c if c.isalnum() or c in ['-', '_'] else '_' for c in base_filename)
            
            s3_base_path = f"{org_id}/{user_id}/{source}/{s3_friendly_folder_name}"
            
            if category == 'unstructured':
                result = await self.unstructured_processor.process_unstructured_document(file_data, file_type, s3_base_path)
            elif category == 'structured':
                result = await self.structured_processor.process_structured_document(file_data, file_type, s3_base_path)
            elif category == 'direct':
                result = await self.direct_processor.process_direct_document(file_data, file_type, s3_base_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            result_with_metadata = {
                "task_id": task_message.task_id,
                "file_type": file_type,
                "category": category,
                "source": source,
                "user_id": user_id,
                "org_id": org_id,
                "s3_url": s3_url,
                "filename": source_filename,
                "internal_object_id": internal_object_id,
                "metadata": payload.get('metadata', {}),
                **result
            }
            
            success = False
            if category == 'unstructured':
                success = await self.neo4j_handler.store_unstructured_document(result_with_metadata)
            elif category == 'structured':
                success = await self.neo4j_handler.store_structured_document(result_with_metadata)
            elif category == 'direct':
                success = await self.neo4j_handler.store_direct_document(result_with_metadata)
            
            if not success:
                logger.error(f"Failed to store document in Neo4j: {source_filename}")
                raise Exception(f"Failed to store document in Neo4j: {source_filename}")
            
            logger.info(f"Successfully processed and stored file with {len(result.get('data', []))} items")
            
            return result_with_metadata
            
        except Exception as e:
            logger.error(f"Error processing file task {task_message.task_id}: {str(e)}")
            raise