import fitz  # PyMuPDF
import magic
import logging

class DocumentProcessor:
    @staticmethod
    def extract_text(file_path: str):
        """Handle PDF/TXT files"""
        try:
            file_type = magic.from_file(file_path, mime=True)
            
            if file_type == "application/pdf":
                return DocumentProcessor._extract_pdf_text(file_path)
            elif file_type == "text/plain":
                with open(file_path, 'r') as f:
                    return f.read()
            else:
                logging.warning(f"Unsupported file type: {file_type}")
                return None

        except Exception as e:
            logging.error(f"Text extraction failed: {str(e)}")
            return None

    @staticmethod
    def _extract_pdf_text(file_path: str):
        """PDF text extraction using PyMuPDF"""
        text = ""
        try:
            doc = fitz.open(file_path)
            for page in doc:
                text += page.get_text()
            return text
        except Exception as e:
            logging.error(f"PDF processing error: {str(e)}")
            return None
