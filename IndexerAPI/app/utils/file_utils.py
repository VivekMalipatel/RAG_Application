import os
import shutil
import logging
from pathlib import Path
from fastapi import UploadFile
from typing import Optional

logger = logging.getLogger(__name__)

async def save_upload_file(upload_file: UploadFile, directory: str, file_id: str) -> str:
    """
    Save uploaded file to a specified directory with a unique name
    
    Args:
        upload_file: FastAPI UploadFile
        directory: Directory to save the file to
        file_id: Unique ID to use for the filename
    
    Returns:
        str: Path to the saved file
    """
    try:
        # Ensure the directory exists
        os.makedirs(directory, exist_ok=True)
        
        # Get file extension
        extension = ""
        if upload_file.filename and "." in upload_file.filename:
            extension = Path(upload_file.filename).suffix
        
        # Create full path with unique ID
        file_path = os.path.join(directory, f"{file_id}{extension}")
        
        # Save the file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
        
        logger.info(f"File saved successfully to {file_path}")
        return file_path
        
    except Exception as e:
        logger.error(f"Error saving file: {str(e)}")
        raise
    finally:
        # Reset file pointer
        await upload_file.seek(0)

def get_file_size(file_path: str) -> Optional[int]:
    """
    Get file size in bytes
    
    Args:
        file_path: Path to the file
    
    Returns:
        int: Size in bytes or None if file doesn't exist
    """
    try:
        return os.path.getsize(file_path)
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return None
    except Exception as e:
        logger.error(f"Error getting file size: {str(e)}")
        return None

def cleanup_temporary_files(directory: str, max_age_hours: int = 24):
    """
    Clean up temporary files older than specified hours
    
    Args:
        directory: Directory to clean
        max_age_hours: Maximum age of files in hours
    """
    try:
        # This is a placeholder function that would be implemented 
        # to remove old temporary files to free up disk space
        logger.info(f"Cleaning up temporary files in {directory} older than {max_age_hours} hours")
        # Implementation would check file modification times and delete old files
        pass
    except Exception as e:
        logger.error(f"Error cleaning up temporary files: {str(e)}")