import io
import os
import subprocess
import asyncio
import logging
import base64
from urllib.parse import urlparse
from pdf2image import convert_from_bytes
from magika import Magika
from core.storage.s3_handler import S3Handler
from config import settings

logger = logging.getLogger(__name__)

def rasterize_and_encode(page_bytes: bytes, page_num: int) -> tuple[int, str]:
    images = convert_from_bytes(page_bytes, dpi=300)
    img = images[0]
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return page_num, b64

async def convert_to_pdf(file_data: bytes, file_type: str) -> bytes:
    if file_type == 'pdf':
        return file_data
        
    try:
        loop = asyncio.get_running_loop()
        
        def _convert_pdf():
            input_bytes = io.BytesIO(file_data)
            input_bytes.seek(0)
            
            temp_input_path = "/tmp/input_document"
            temp_output_path = "/tmp/output_document.pdf"
            
            with open(temp_input_path, 'wb') as temp_file:
                temp_file.write(input_bytes.getvalue())
            
            cmd = [
                "unoconvert",
                "--host-location", "remote",
                "--host", settings.UNOSERVER_HOST,
                "--port", settings.UNOSERVER_PORT,
                temp_input_path,
                temp_output_path
            ]
            
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            with open(temp_output_path, 'rb') as output_file:
                pdf_data = output_file.read()
            
            if os.path.exists(temp_input_path):
                os.remove(temp_input_path)
            if os.path.exists(temp_output_path):
                os.remove(temp_output_path)
            
            return pdf_data
        
        return await loop.run_in_executor(None, _convert_pdf)
    except Exception as e:
        logger.error(f"Error converting file to PDF: {e}")
        raise ValueError(f"Failed to convert file to PDF: {str(e)}")

async def detect_file_type(file_data: bytes) -> str:
    try:
        loop = asyncio.get_running_loop()
        magika = Magika()
        file_stream = io.BytesIO(file_data)
        kind = await loop.run_in_executor(None, magika.identify_stream, file_stream)
        if kind is not None:
            return kind.output.label
        raise ValueError("Could not detect file type")
    except Exception as e:
        logger.error(f"Error detecting file type: {e}")
        return ''

async def download_file_from_s3_url(s3_url: str) -> bytes:
    try:
        parsed_url = urlparse(s3_url)
        s3_handler = S3Handler()
        
        if parsed_url.scheme in ['s3']:
            bucket_name = parsed_url.netloc
            object_key = parsed_url.path.lstrip('/')
        else:
            path_parts = parsed_url.path.strip('/').split('/')
            if len(path_parts) < 2:
                raise ValueError(f"Invalid S3 URL format: {s3_url}")
            bucket_name = path_parts[0]
            object_key = '/'.join(path_parts[1:])
        
        logger.info(f"Downloading file from S3: bucket={bucket_name}, key={object_key}")
        
        if bucket_name != s3_handler.bucket_name:
            logger.warning(f"S3 URL bucket ({bucket_name}) differs from configured bucket ({s3_handler.bucket_name})")
        
        file_data = await s3_handler.download_bytes(object_key)
        logger.info(f"Successfully downloaded {len(file_data)} bytes from S3")
        return file_data
        
    except Exception as e:
        logger.error(f"Error downloading file from S3 URL {s3_url}: {e}")
        raise ValueError(f"Failed to download file from S3: {str(e)}")
