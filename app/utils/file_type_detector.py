import magic
import logging

class FileTypeDetector:
    """Utility class to detect file MIME types."""
    
    @staticmethod
    def get_mime_type(file_path: str):
        """Returns the MIME type of a given file."""
        try:
            mime = magic.Magic(mime=True)
            file_type = mime.from_file(file_path)
            return file_type
        except Exception as e:
            logging.error(f"File type detection failed for {file_path}: {str(e)}")
            return None