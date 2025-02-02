import asyncio
import fitz  # PyMuPDF
from unstructured.partition.auto import partition

class TextProcessor:
    """Extracts text from PDFs and DOCX files asynchronously."""

    async def extract_pdf_text(self, pdf_path: str):
        """Asynchronously extracts text from a PDF file."""
        return await asyncio.to_thread(self._extract_pdf_sync, pdf_path)

    def _extract_pdf_sync(self, pdf_path: str):
        """Sync PDF text extraction wrapped for async."""
        doc = fitz.open(pdf_path)
        return "\n".join(page.get_text() for page in doc)

    async def extract_docx_text(self, docx_path: str):
        """Asynchronously extracts text from a DOCX file using Unstructured AI."""
        elements = await asyncio.to_thread(partition, filename=docx_path)
        return "\n".join(str(el) for el in elements)