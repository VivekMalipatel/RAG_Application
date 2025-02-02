import pytesseract
from PIL import Image
import os

class ImageProcessor:
    """Extracts text from images using Tesseract OCR."""

    def __init__(self):
        """Ensure Tesseract is correctly configured."""
        self.tesseract_cmd = os.getenv("TESSERACT_CMD", "/usr/bin/tesseract")
        pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd

    def extract_text(self, image_path: str):
        """Extracts text from an image using OCR."""
        try:
            image = Image.open(image_path)
            return pytesseract.image_to_string(image).strip()
        except Exception as e:
            print(f"🔹 Image OCR Error: {e}")
            return ""