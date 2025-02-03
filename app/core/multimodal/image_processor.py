import pytesseract
from PIL import Image
import os
import logging

class ImageProcessor:
    """Extracts text from images using Tesseract OCR."""

    def __init__(self):
        """Ensure Tesseract is correctly configured."""
        self.tesseract_cmd = os.getenv("TESSERACT_CMD", "/usr/bin/tesseract")
        pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd
        logging.info(f"Tesseract OCR configured at: {self.tesseract_cmd}")

    def extract_text(self, image_path: str) -> str:
        """
        Extracts text from an image using OCR.

        Args:
            image_path (str): Path to the image file.

        Returns:
            str: Extracted text content.
        """
        try:
            with Image.open(image_path) as image:
                extracted_text = pytesseract.image_to_string(image).strip()
            logging.info(f"Successfully extracted text from image: {image_path}")
            return extracted_text
        except FileNotFoundError:
            logging.error(f"Image file not found: {image_path}")
        except Exception as e:
            logging.error(f"Error processing image '{image_path}': {e}")
        return ""