import fitz  # PyMuPDF
import os

class TextProcessor:
    """Extracts text from PDFs and DOCX files."""

    def extract_text(self, file_path: str):
        """Handles PDF text extraction."""
        text = ""
        try:
            doc = fitz.open(file_path)
            for page in doc:
                text += page.get_text()
        except Exception as e:
            print(f"🔹 PDF Processing Error: {e}")
        return text.strip()