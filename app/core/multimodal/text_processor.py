import fitz  # PyMuPDF
import logging

class TextProcessor:
    """Extracts text from PDF files."""

    def extract_text(self, file_path: str) -> str:
        """
        Extracts text from a PDF file.

        Args:
            file_path (str): Path to the PDF file.

        Returns:
            str: Extracted text content.
        """
        text = ""
        try:
            with fitz.open(file_path) as doc:
                text = "\n".join(page.get_text() for page in doc)
            logging.info(f"Successfully extracted text from {file_path}")
        except Exception as e:
            logging.error(f"Error processing PDF file '{file_path}': {e}")
        return text.strip()