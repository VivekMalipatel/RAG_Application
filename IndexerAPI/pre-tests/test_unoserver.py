#!/usr/bin/env python3

import os
import sys
import logging
import io
from pathlib import Path

# Add the project root to the Python path to allow imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from processors.file_processor import FileProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_unoserver")

def save_pdf(pdf_bytes: bytes, output_path: str) -> None:
    """Save PDF bytes to a file"""
    with open(output_path, 'wb') as f:
        f.write(pdf_bytes)
    logger.info(f"Saved PDF to {output_path}")

def main():
    """Test converting different file types to PDF using unoserver"""
    logger.info("Starting PDF conversion test")
    
    # Initialize the FileProcessor
    processor = FileProcessor()
    
    # Test files
    test_files = [
        {"path": "pre-tests/Pokemon.csv", "type": "csv"},
        {"path": "pre-tests/Resume.md", "type": "markdown"},
        {"path": "pre-tests/Vivek Malipatel - Resume.docx", "type": "docx"}
    ]
    
    # Create output directory if it doesn't exist
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    
    for test_file in test_files:
        file_path = test_file["path"]
        file_type = test_file["type"]
        output_path = output_dir / f"{Path(file_path).stem}.pdf"
        
        try:
            logger.info(f"Converting {file_path} to PDF")
            
            # Read the file
            with open(file_path, 'rb') as f:
                file_data = f.read()
            
            # Convert to PDF
            pdf_bytes = processor.convert_to_pdf(file_data, file_type)
            
            # Save the output
            save_pdf(pdf_bytes, output_path)
            logger.info(f"Successfully converted {file_path} to PDF")
            
        except Exception as e:
            logger.error(f"Error converting {file_path} to PDF: {e}")
    
    logger.info("PDF conversion test completed")

if __name__ == "__main__":
    main()