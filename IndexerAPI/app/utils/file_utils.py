import os
import shutil
import logging
from pathlib import Path
from fastapi import UploadFile
from typing import Optional

logger = logging.getLogger(__name__)

async def save_upload_file(upload_file: UploadFile, directory: str, file_id: str) -> str:
    try:
        os.makedirs(directory, exist_ok=True)
        
        extension = ""
        if upload_file.filename and "." in upload_file.filename:
            extension = Path(upload_file.filename).suffix
        
        file_path = os.path.join(directory, f"{file_id}{extension}")
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
        
        logger.info(f"File saved successfully to {file_path}")
        return file_path
        
    except Exception as e:
        logger.error(f"Error saving file: {str(e)}")
        raise
    finally:
        await upload_file.seek(0)

def get_file_size(file_path: str) -> Optional[int]:
    try:
        return os.path.getsize(file_path)
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return None
    except Exception as e:
        logger.error(f"Error getting file size: {str(e)}")
        return None

def cleanup_temporary_files(directory: str, max_age_hours: int = 24):
    try:
        logger.info(f"Cleaning up temporary files in {directory} older than {max_age_hours} hours")
        pass
    except Exception as e:
        logger.error(f"Error cleaning up temporary files: {str(e)}")