#!/usr/bin/env python
import os
import logging
from pathlib import Path
from markitdown import MarkItDown
import tempfile

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_basic_conversion():
    """Test basic file conversion with MarkItDown"""
    logger.info("Testing basic MarkItDown conversion")
    
    # Initialize MarkItDown (plugins disabled by default)
    md = MarkItDown(enable_plugins=False)
    
    # Create a temporary text file for testing
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False, mode='w') as temp:
        temp.write("""# Sample Document
        
## Section 1
This is a test document for MarkItDown.
        
## Section 2
* List item 1
* List item 2
* List item 3
        
## Section 3
This is a [link to Microsoft](https://microsoft.com).

Here's a simple table:
| Name | Age | Role |
|------|-----|------|
| John | 30  | Dev  |
| Jane | 28  | PM   |
""")
        temp_path = temp.name
    
    try:
        # Convert the text file
        logger.info(f"Converting file: {temp_path}")
        result = md.convert(temp_path)
        
        # Display result
        logger.info("Conversion successful!")
        logger.info("Converted content:")
        print("-" * 50)
        print(result.text_content)
        print("-" * 50)
        
        # Check if the result has specific attributes
        for attr in dir(result):
            if not attr.startswith('_'):
                logger.info(f"Available attribute: {attr}")
                
    except Exception as e:
        logger.error(f"Error during conversion: {str(e)}")
        import traceback
        logger.error(f"Error details: {traceback.format_exc()}")
    finally:
        # Clean up the temporary file
        os.unlink(temp_path)

def test_convert_multiple_formats():
    """Test converting multiple file formats if available"""
    logger.info("Testing conversion of different file formats")
    
    # Initialize MarkItDown with plugins enabled
    md = MarkItDown(enable_plugins=True)
    
    # Check for available converters
    logger.info("MarkItDown instance information:")
    for attr in dir(md):
        if not attr.startswith('_'):
            logger.info(f"- Attribute: {attr}")
    
    # List sample files to test conversion (if they exist)
    sample_files = [
        './samples/sample.pdf',
        './samples/sample.docx',
        './samples/sample.pptx',
        './samples/sample.jpg',
    ]
    
    # Create samples directory if it doesn't exist
    Path('./samples').mkdir(exist_ok=True)
    
    # If no sample files exist, create a simple one
    if not any(Path(f).exists() for f in sample_files):
        logger.info("No sample files found, creating a simple text file")
        with open('./samples/sample.txt', 'w') as f:
            f.write("This is a test sample file for MarkItDown.")
        sample_files.append('./samples/sample.txt')
    
    for file_path in sample_files:
        path = Path(file_path)
        if path.exists():
            try:
                logger.info(f"Converting {path}...")
                result = md.convert(str(path))
                logger.info(f"Successfully converted {path.name}")
                logger.info(f"First 500 chars: {result.text_content[:500]}...")
            except Exception as e:
                logger.error(f"Failed to convert {path}: {str(e)}")

def convert_pdf_to_markdown(pdf_path, output_path=None):
    """Convert a PDF file to markdown and optionally save to a file"""
    logger.info(f"Converting PDF to markdown: {pdf_path}")
    
    # Initialize MarkItDown with plugins enabled to handle PDFs
    md = MarkItDown(enable_plugins=True)
    
    try:
        # Convert the PDF file
        result = md.convert(pdf_path)
        
        # Display conversion info
        logger.info(f"Successfully converted {pdf_path}")
        logger.info(f"Content length: {len(result.text_content)} characters")
        
        # Save to file if output path is provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result.text_content)
            logger.info(f"Saved markdown content to {output_path}")
        
        return result.text_content
    except Exception as e:
        logger.error(f"Error converting {pdf_path}: {str(e)}")
        import traceback
        logger.error(f"Error details: {traceback.format_exc()}")
        return None

def main():
    """Main function to process the PDF files"""
    logger.info("Starting PDF to Markdown conversion")
    
    # Get absolute paths for the PDF files
    base_dir = Path('/mainpool/Projects/RAG_Application/Application/IndexerAPI')
    
    pdf_files = {
        'resume': base_dir / 'pre-tests/Resume.pdf',
        'ragas': base_dir / 'pre-tests/ragas_papers.pdf',
        'csv': base_dir / 'pre-tests/Pokemon.csv'
    }
    
    # Create output paths for markdown files
    md_files = {
        'resume': base_dir / 'pre-tests/Resume.md',
        'ragas': base_dir / 'pre-tests/ragas_papers.md',
        'csv': base_dir / 'pre-tests/Pokemon.md'
    }
    
    # Process each PDF
    for name, pdf_path in pdf_files.items():
        if pdf_path.exists():
            logger.info(f"Processing {name} PDF: {pdf_path}")
            markdown_content = convert_pdf_to_markdown(str(pdf_path), str(md_files[name]))
            
            # Print first 500 characters of converted content
            if markdown_content:
                preview = markdown_content[:500] + "..." if len(markdown_content) > 500 else markdown_content
                logger.info(f"Preview of {name} markdown content:\n{preview}")
        else:
            logger.warning(f"{name} PDF not found at {pdf_path}")
    
    logger.info("Conversion complete")

if __name__ == "__main__":
    main()