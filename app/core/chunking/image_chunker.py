import cv2
import numpy as np
import pytesseract
import asyncio

class ImageChunker:
    """Splits images into smaller tiles and extracts text using OCR."""

    def __init__(self, tile_size=512):
        self.tile_size = tile_size

    async def chunk_image(self, image_path: str):
        """Splits an image into smaller tiles for OCR processing."""
        return await asyncio.to_thread(self._chunk_image_sync, image_path)

    def _chunk_image_sync(self, image_path: str):
        """Sync function wrapped for async execution."""
        image = cv2.imread(image_path)
        h, w, _ = image.shape
        chunks = []

        for y in range(0, h, self.tile_size):
            for x in range(0, w, self.tile_size):
                tile = image[y:y+self.tile_size, x:x+self.tile_size]
                text = pytesseract.image_to_string(tile)
                if text.strip():
                    chunks.append(text)

        return chunks