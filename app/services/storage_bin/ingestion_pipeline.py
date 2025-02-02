from app.core.storage_bin.minio import MinIOClient
from document_processor import DocumentProcessor
import tempfile
import os

class IngestionPipeline:
    def __init__(self):
        self.storage = MinIOClient()
        self.processor = DocumentProcessor()

    def process_upload(self, user_id: str, file_data: bytes, file_name: str):
        """Main processing workflow"""
        try:
            # 1. Store raw file
            temp_path = self._save_temp_file(file_data, file_name)
            
            # 2. Create user bucket
            self.storage.create_user_bucket(user_id)
            
            # 3. Extract text content
            text_content = self.processor.extract_text(temp_path)
            
            # 4. Cleanup
            os.remove(temp_path)
            
            return text_content
            
        except Exception as e:
            print(f"Pipeline error: {str(e)}")
            return None

    def _save_temp_file(self, data: bytes, filename: str):
        """Save to temporary storage"""
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, filename)
        with open(file_path, 'wb') as f:
            f.write(data)
        return file_path

