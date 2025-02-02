import pytesseract
from pdf2image import convert_from_path
import cv2
import numpy as np
import asyncio

class ImageProcessor:
    """Extracts text from images using OCR."""

    async def extract_text_from_image(self, image_path: str):
        """Asynchronously extracts text from an image."""
        return await asyncio.to_thread(self._extract_image_text_sync, image_path)

    def _extract_image_text_sync(self, image_path: str):
        """Sync function wrapped for async execution."""
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        processed_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        return pytesseract.image_to_string(processed_image)

    async def extract_text_from_pdf_images(self, pdf_path: str):
        """Extract text from images inside a PDF."""
        images = await asyncio.to_thread(convert_from_path, pdf_path)
        text_results = []
        for image in images:
            text = await self.extract_text_from_image(image)
            text_results.append(text)
        return "\n".join(text_results)