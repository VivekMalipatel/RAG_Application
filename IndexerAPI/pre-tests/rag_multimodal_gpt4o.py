#!/usr/bin/env python
"""
RAG Over PDFs with ColNomic Embed Multimodal and GPT-4o API

This script implements a multimodal retrieval-augmented generation system 
for querying PDF documents using ColNomic Embed for retrieval and OpenAI's GPT-4o for generation.
It provides a conversational interface with streaming responses.
"""

# Set tokenizers parallelism to False to avoid warnings
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import base64
import io
import json
import time
import sys
import matplotlib.pyplot as plt
from pdf2image import convert_from_path
from PIL import Image
import requests
import torch
from typing import Dict, List, Iterator, Optional, Union

# Set up custom directory for model storage (absolute path)
current_file = os.path.abspath(__file__)
models_dir = os.path.join(os.path.dirname(current_file), ".models")
os.makedirs(models_dir, exist_ok=True)

# Set environment variables to force HF to use our custom cache directory
os.environ["HF_HOME"] = models_dir
os.environ["TRANSFORMERS_CACHE"] = os.path.join(models_dir, "transformers")
os.environ["HF_DATASETS_CACHE"] = os.path.join(models_dir, "datasets")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(models_dir, "hub")

# Check if CUDA is available
device = "cuda:0" if torch.cuda.is_available() else "mps"
print(f"Using device: {device}")
print(f"Models will be stored at: {models_dir}")
print(f"HF_HOME set to: {os.environ['HF_HOME']}")
print(f"TRANSFORMERS_CACHE set to: {os.environ['TRANSFORMERS_CACHE']}")

# Define paths to PDFs
RESUME_PDF = "/Users/vivekmalipatel/Downloads/Test/Resume.pdf"
RAGAS_PAPER_PDF = "/Users/vivekmalipatel/Downloads/Test/ragas_papers.pdf"

# Set your OpenAI API key here if not set in environment
# OPENAI_API_KEY = "your-api-key-here"

# Color codes for terminal output
class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    ENDC = '\033[0m'

def print_colored(text, color=Colors.BLUE, end='\n'):
    """Print colored text to terminal."""
    print(f"{color}{text}{Colors.ENDC}", end=end)
    sys.stdout.flush()

def display_pdf_images(images_list, title="PDF Preview"):
    """Display all images in the provided list as subplots with 5 images per row."""
    num_images = len(images_list)
    num_rows = num_images // 5 + (1 if num_images % 5 > 0 else 0)
    
    # Handle the case of a single image
    if num_images == 1:
        plt.figure(figsize=(8, 8))
        plt.imshow(images_list[0])
        plt.title(f"{title} - Page 1")
        plt.axis('off')
    else:
        fig, axes = plt.subplots(num_rows, min(5, num_images), figsize=(20, 4 * num_rows))
        if num_rows == 1 and num_images < 5:
            axes = [axes] if num_images == 1 else axes
            
        # Convert to flatten array if necessary
        if num_rows > 1 or num_images >= 5:
            axes = axes.flatten()
            
        for i, img in enumerate(images_list):
            if i < len(axes):
                ax = axes[i]
                ax.imshow(img)
                ax.set_title(f"{title} - Page {i+1}")
                ax.axis('off')
                
        # Hide empty subplots
        if num_images < len(axes):
            for j in range(num_images, len(axes)):
                axes[j].axis('off')
                
        plt.tight_layout()
    
    plt.show()

def create_rag_model():
    """Initialize the RAG model with ColNomic Embed Multimodal."""
    from byaldi import RAGMultiModalModel
    
    print_colored("Loading ColNomic Embed Multimodal retrieval model...", Colors.YELLOW)
    
    # Initialize the model with minimal parameters
    print_colored(f"Models will be stored at: {models_dir}", Colors.YELLOW)
    
    try:
        # Use only the device parameter which is known to work
        rag = RAGMultiModalModel.from_pretrained(
            "nomic-ai/colnomic-embed-multimodal-3b",
            device=device
        )
        print_colored("Retrieval model loaded successfully!", Colors.GREEN)
        return rag
    except Exception as e:
        print_colored(f"Error during model loading: {str(e)}", Colors.RED)
        print_colored("Trying with default parameters...", Colors.YELLOW)
        # Fall back to default parameters if specific ones fail
        rag = RAGMultiModalModel.from_pretrained(
            "nomic-ai/colnomic-embed-multimodal-3b"
        )
        print_colored("Retrieval model loaded successfully with default parameters!", Colors.GREEN)
        return rag

def encode_image_to_base64(image):
    """Encode PIL image to base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def stream_query_gpt4o(query: str, images: List[Image], conversation_history: List[Dict] = None, api_key: str = None) -> Iterator[str]:
    """
    Stream responses from GPT-4o with text and images.
    
    Args:
        query: The text query
        images: List of PIL images
        conversation_history: Previous conversation messages
        api_key: OpenAI API key (if not set in environment)
    
    Yields:
        Chunks of the generated response text
    """
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        yield "Error: OpenAI API key not found. Please set your API key."
        return
    
    # Prepare the API request
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # Encode all images to base64
    image_content = []
    for img in images:
        base64_image = encode_image_to_base64(img)
        image_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        })
    
    # Prepare the user message with images and query
    user_content = [
        *image_content,
        {"type": "text", "text": query}
    ]
    
    # Initialize conversation history if not provided
    if conversation_history is None:
        conversation_history = [
            {
                "role": "system",
                "content": "You are an expert PDF analyst. Analyze the provided PDF pages and answer questions thoroughly based solely on the visible content. Be concise but comprehensive. If the answer isn't in the provided pages, say so."
            }
        ]
    
    # Add the new user message
    conversation_history.append({"role": "user", "content": user_content})
    
    # Create the request payload
    payload = {
        "model": "gpt-4o",
        "messages": conversation_history,
        "max_tokens": 1000,
        "stream": True
    }
    
    # Define retry parameters
    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            # Make streaming request
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                stream=True,
                timeout=60  # Add timeout to prevent hanging
            )
            
            # Handle streaming response
            if response.status_code == 200:
                collected_messages = []
                
                # Process the streaming response
                for line in response.iter_lines():
                    if not line:
                        continue
                    
                    # Remove "data: " prefix
                    if line.startswith(b'data: '):
                        line = line[6:]
                    
                    if line.strip() == b'[DONE]':
                        break
                    
                    try:
                        json_line = json.loads(line)
                        content = json_line.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        if content:
                            collected_messages.append(content)
                            yield content
                    except Exception as e:
                        yield f"\nError parsing stream: {str(e)}\n"
                
                # Add the assistant response to conversation history
                conversation_history.append({
                    "role": "assistant",
                    "content": "".join(collected_messages)
                })
                
                # If we get here, request was successful, so break the retry loop
                break
            elif response.status_code in [429, 500, 502, 503, 504]:
                # Rate limit or server error - retry
                if attempt < max_retries - 1:  # Don't delay on the last attempt
                    yield f"\nAPI request failed with status {response.status_code}. Retrying in {retry_delay} seconds..."
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    error_message = f"Error after {max_retries} attempts: API request failed with status code {response.status_code}."
                    yield error_message
            else:
                error_message = f"Error: API request failed with status code {response.status_code}. {response.text}"
                yield error_message
                break  # Don't retry for client errors
        
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                yield f"\nRequest timed out. Retrying in {retry_delay} seconds..."
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                yield f"Error: Request timed out after {max_retries} attempts."
        
        except requests.exceptions.RequestException as e:
            yield f"Network error: {str(e)}"
            break
        
        except Exception as e:
            yield f"Error: {str(e)}"
            break

def process_pdf(pdf_path, rag_model, index_name):
    """Process a PDF file: convert to images, display preview, and index it for RAG."""
    print_colored(f"Processing PDF: {os.path.basename(pdf_path)}", Colors.YELLOW)
    
    # Define the index path
    index_path = os.path.join(os.path.dirname(current_file), ".byaldi", index_name)
    
    # Convert PDF to images regardless of whether we need to reindex
    try:
        pdf_images = convert_from_path(pdf_path)
        print_colored(f"Successfully converted PDF to {len(pdf_images)} images", Colors.GREEN)
        
        # Display a preview of the PDF (first few pages)
        max_preview = min(2, len(pdf_images))
        display_pdf_images(pdf_images[:max_preview], title=os.path.basename(pdf_path))
        
        # Check if index already exists
        if os.path.exists(index_path):
            print_colored(f"Index for {os.path.basename(pdf_path)} already exists. Skipping indexing.", Colors.GREEN)
            # Try to load the existing index
            try:
                rag_model.load_index(index_name)
                print_colored(f"Successfully loaded existing index: {index_name}", Colors.GREEN)
            except Exception as e:
                print_colored(f"Error loading existing index: {e}", Colors.RED)
                print_colored("Will regenerate the index.", Colors.YELLOW)
                rag_model.index(
                    input_path=pdf_path,
                    index_name=index_name,
                    store_collection_with_index=False,
                    overwrite=True
                )
                print_colored(f"Indexing complete for {os.path.basename(pdf_path)}", Colors.GREEN)
        else:
            # Index the PDF for RAG if it doesn't exist
            print_colored(f"Indexing {os.path.basename(pdf_path)} for retrieval...", Colors.YELLOW)
            rag_model.index(
                input_path=pdf_path,
                index_name=index_name,
                store_collection_with_index=False,
                overwrite=True
            )
            print_colored(f"Indexing complete for {os.path.basename(pdf_path)}", Colors.GREEN)
        
        return pdf_images
        
    except Exception as e:
        print_colored(f"Error processing PDF: {e}", Colors.RED)
        return None

def handle_query(query: str, rag_model, pdf_name: str, pdf_images: List[Image], 
                conversation_history: List[Dict], k: int = 2, api_key: str = None) -> List[Dict]:
    """
    Handle a user query for a specific PDF.
    
    Args:
        query: User query text
        rag_model: The RAG model instance
        pdf_name: Name of the PDF being queried
        pdf_images: List of PDF page images
        conversation_history: Conversation history
        k: Number of pages to retrieve
        api_key: OpenAI API key
    
    Returns:
        Updated conversation history
    """
    print_colored(f"\nRetrieving information from {pdf_name}...", Colors.YELLOW)
    
    # Search relevant pages
    rag_results = rag_model.search(query, k=k)
    
    if not rag_results:
        print_colored("No relevant pages found for your query.", Colors.RED)
        return conversation_history
    
    # Get the images for the retrieved pages
    retrieved_images = [pdf_images[result["page_num"] - 1] for result in rag_results]
    
    # Print which pages were retrieved
    pages = [result["page_num"] for result in rag_results]
    print_colored(f"Found relevant content on page(s): {', '.join(map(str, pages))}", Colors.GREEN)
    
    # Stream answer from GPT-4o
    print_colored("\nGenerating response: ", Colors.BOLD, end='')
    
    # Keep track of streamed tokens to avoid duplicates
    for chunk in stream_query_gpt4o(query, retrieved_images, conversation_history, api_key):
        print(chunk, end='')
        sys.stdout.flush()
    
    print("\n")  # Add newline after response
    
    return conversation_history

def interactive_mode(document_data, api_key=None):
    """
    Run the RAG system in interactive conversation mode.
    
    Args:
        document_data: Dictionary mapping document names to their page images and models
        api_key: OpenAI API key
    """
    current_document = None
    conversation_histories = {doc_name: [] for doc_name in document_data.keys()}
    
    # Display welcome message
    print_colored("\n" + "="*80, Colors.BOLD)
    print_colored(" 📑 Welcome to the Interactive PDF RAG System with GPT-4o! 🤖", Colors.BOLD)
    print_colored("="*80, Colors.BOLD)
    print_colored("\nAvailable documents:", Colors.YELLOW)
    for i, doc_name in enumerate(document_data.keys(), 1):
        print_colored(f"  {i}. {doc_name}", Colors.GREEN)
    
    # Instructions
    print_colored("\nCommands:", Colors.YELLOW)
    print_colored("  !help - Show this help message", Colors.BLUE)
    print_colored("  !switch <doc_number> - Switch to another document", Colors.BLUE)
    print_colored("  !quit or !exit - Exit the program", Colors.BLUE)
    
    if not current_document:
        first_doc = list(document_data.keys())[0]
        current_document = first_doc
        print_colored(f"\nCurrently using document: {current_document}", Colors.GREEN)
    
    # Main conversation loop
    while True:
        # Get user input
        try:
            print_colored("\n> ", Colors.BOLD, end='')
            user_input = input().strip()
            
            # Check for commands
            if user_input.lower() in ["!quit", "!exit", "quit", "exit"]:
                print_colored("\nThank you for using the PDF RAG System! Goodbye!", Colors.GREEN)
                break
                
            elif user_input.lower() == "!help":
                print_colored("\nCommands:", Colors.YELLOW)
                print_colored("  !help - Show this help message", Colors.BLUE)
                print_colored("  !switch <doc_number> - Switch to another document", Colors.BLUE)
                print_colored("  !quit or !exit - Exit the program", Colors.BLUE)
                print_colored(f"\nCurrently using document: {current_document}", Colors.GREEN)
                continue
                
            elif user_input.lower().startswith("!switch"):
                try:
                    doc_num = int(user_input.split()[1]) - 1
                    if 0 <= doc_num < len(document_data):
                        current_document = list(document_data.keys())[doc_num]
                        print_colored(f"\nSwitched to document: {current_document}", Colors.GREEN)
                    else:
                        print_colored("\nInvalid document number. Please try again.", Colors.RED)
                except (IndexError, ValueError):
                    print_colored("\nUsage: !switch <doc_number>", Colors.RED)
                continue
                
            elif not user_input:
                continue
                
            # Process the query
            current_history = conversation_histories[current_document]
            conversation_histories[current_document] = handle_query(
                user_input,
                document_data[current_document]["model"],
                current_document,
                document_data[current_document]["images"],
                current_history,
                k=2,
                api_key=api_key
            )
            
        except KeyboardInterrupt:
            print_colored("\n\nInterrupted. Enter !quit to exit or continue with a new query.", Colors.YELLOW)
        except Exception as e:
            print_colored(f"\nError: {str(e)}", Colors.RED)

def create_unified_index(rag_model, index_name, images):
    """
    Create a unified index from a list of images.
    This allows us to index multiple documents as a single collection.
    """
    print_colored("Creating unified index from all documents...", Colors.YELLOW)
    
    # Create a temporary directory to save images
    import tempfile
    import os
    from PIL import Image
    
    # Check if the index already exists
    index_path = os.path.join(os.path.dirname(current_file), ".byaldi", index_name)
    if os.path.exists(index_path):
        print_colored("Unified index already exists. Will use it automatically when searching.", Colors.GREEN)
        # Note: No need to load the index explicitly - it will be loaded automatically when search() is called
        return
    
    # Since ColPaliModel doesn't accept a list of paths, we'll create a temporary PDF
    print_colored("Creating temporary document for indexing...", Colors.YELLOW)
    
    # Convert all images to a single PDF
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_pdf:
        temp_pdf_path = temp_pdf.name
        
    # Save the images as a single PDF
    images[0].save(
        temp_pdf_path, 
        save_all=True, 
        append_images=images[1:] if len(images) > 1 else []
    )
    
    print_colored(f"Indexing {len(images)} pages from temporary PDF...", Colors.YELLOW)
    
    # Now index the temporary PDF
    rag_model.index(
        input_path=temp_pdf_path,
        index_name=index_name,
        store_collection_with_index=False,
        overwrite=True
    )
    
    # Clean up
    os.unlink(temp_pdf_path)
    
    print_colored("Unified indexing complete!", Colors.GREEN)

def interactive_unified_mode(rag_model, all_images, page_mapping, api_key=None):
    """
    Run the RAG system in interactive conversation mode with unified document pool.
    
    Args:
        rag_model: The unified RAG model
        all_images: All PDF page images combined
        page_mapping: Mapping from combined index to original doc/page
        api_key: OpenAI API key
    """
    conversation_history = []
    
    # Display welcome message
    print_colored("\n" + "="*80, Colors.BOLD)
    print_colored(" 📑 Welcome to the Unified PDF RAG System with GPT-4o! 🤖", Colors.BOLD)
    print_colored("="*80, Colors.BOLD)
    
    # Show document stats
    docs = set(info["document"] for info in page_mapping.values())
    print_colored(f"\nCombined index contains {len(docs)} documents with {len(all_images)} total pages", Colors.GREEN)
    for doc in docs:
        count = sum(1 for info in page_mapping.values() if info["document"] == doc)
        print_colored(f"  • {doc}: {count} pages", Colors.BLUE)
    
    # Instructions
    print_colored("\nCommands:", Colors.YELLOW)
    print_colored("  !help - Show this help message", Colors.BLUE)
    print_colored("  !quit or !exit - Exit the program", Colors.BLUE)
    
    # Main conversation loop
    while True:
        # Get user input
        try:
            print_colored("\n> ", Colors.BOLD, end='')
            user_input = input().strip()
            
            # Check for commands
            if user_input.lower() in ["!quit", "!exit", "quit", "exit"]:
                print_colored("\nThank you for using the Unified PDF RAG System! Goodbye!", Colors.GREEN)
                break
                
            elif user_input.lower() == "!help":
                print_colored("\nCommands:", Colors.YELLOW)
                print_colored("  !help - Show this help message", Colors.BLUE)
                print_colored("  !quit or !exit - Exit the program", Colors.BLUE)
                continue
                
            elif not user_input:
                continue
                
            # Process the query using the unified index
            print_colored(f"\nSearching across all documents...", Colors.YELLOW)
            
            # Search relevant pages
            k = 3  # Retrieve more pages for better context
            rag_results = rag_model.search(user_input, k=k)
            
            if not rag_results:
                print_colored("No relevant pages found for your query.", Colors.RED)
                continue
            
            # Get the images for the retrieved pages
            retrieved_images = []
            source_info = []
            
            for result in rag_results:
                page_idx = result["page_num"] - 1
                
                # Add the image
                retrieved_images.append(all_images[page_idx])
                
                # Map back to original document information
                if page_idx in page_mapping:
                    info = page_mapping[page_idx]
                    source_info.append(f"{info['document']} (Page {info['display_page']})")
                else:
                    source_info.append(f"Unknown source (Page {result['page_num']})")
            
            # Print which pages were retrieved with document names
            print_colored(f"Found relevant content from:", Colors.GREEN)
            for source in source_info:
                print_colored(f"  • {source}", Colors.BLUE)
            
            # Stream answer from GPT-4o
            print_colored("\nGenerating response: ", Colors.BOLD, end='')
            
            # Prepare a system message that mentions the source documents
            if not conversation_history:
                conversation_history = [{
                    "role": "system",
                    "content": "You are an expert document analyst. You're analyzing pages from multiple documents. "
                               "For each query, answer thoroughly based solely on the visible content. "
                               "When referring to the source, mention which document and page number the information comes from. "
                               "Be concise but comprehensive. If the answer isn't in the provided pages, say so."
                }]
            
            # Stream the response
            for chunk in stream_query_gpt4o(user_input, retrieved_images, conversation_history, api_key):
                print(chunk, end='')
                sys.stdout.flush()
            
            print("\n")  # Add newline after response
            
        except KeyboardInterrupt:
            print_colored("\n\nInterrupted. Enter !quit to exit or continue with a new query.", Colors.YELLOW)
        except Exception as e:
            print_colored(f"\nError: {str(e)}", Colors.RED)

def main():
    """Main function to process PDFs and start the interactive conversational mode."""
    print_colored("\n🚀 Starting Multimodal RAG System with GPT-4o Streaming...", Colors.BOLD)
    
    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        api_key = input("Please enter your OpenAI API key: ").strip()
        if not api_key:
            print_colored("Error: OpenAI API key is required.", Colors.RED)
            return
        os.environ["OPENAI_API_KEY"] = api_key
    
    # Create a single RAG model for all documents
    try:
        unified_rag_model = create_rag_model()
    except Exception as e:
        print_colored(f"Failed to initialize RAG model: {e}", Colors.RED)
        return
    
    # Process and index PDFs using a single model
    document_images = {}
    all_pdf_images = []
    pdf_page_mapping = {}  # To map page indices to original documents
    current_page_offset = 0
    
    # Define a unified index name
    unified_index_name = "unified_documents_index"
    
    # Process all PDFs first to get their images
    if os.path.exists(RESUME_PDF):
        try:
            resume_images = convert_from_path(RESUME_PDF)
            print_colored(f"Successfully converted Resume.pdf to {len(resume_images)} images", Colors.GREEN)
            document_images["Resume"] = resume_images
            
            # Track which pages belong to Resume
            for i in range(len(resume_images)):
                pdf_page_mapping[current_page_offset + i] = {
                    "document": "Resume",
                    "original_page": i,
                    "display_page": i + 1  # For display (1-indexed)
                }
            
            all_pdf_images.extend(resume_images)
            current_page_offset += len(resume_images)
        except Exception as e:
            print_colored(f"Failed to process Resume PDF: {e}", Colors.RED)
    
    if os.path.exists(RAGAS_PAPER_PDF):
        try:
            ragas_images = convert_from_path(RAGAS_PAPER_PDF)
            print_colored(f"Successfully converted ragas_papers.pdf to {len(ragas_images)} images", Colors.GREEN)
            document_images["RAGAS Paper"] = ragas_images
            
            # Track which pages belong to RAGAS paper
            for i in range(len(ragas_images)):
                pdf_page_mapping[current_page_offset + i] = {
                    "document": "RAGAS Paper",
                    "original_page": i,
                    "display_page": i + 1  # For display (1-indexed)
                }
            
            all_pdf_images.extend(ragas_images)
            current_page_offset += len(ragas_images)
        except Exception as e:
            print_colored(f"Failed to process RAGAS Paper PDF: {e}", Colors.RED)
    
    if not all_pdf_images:
        print_colored("Error: No documents were successfully processed.", Colors.RED)
        return
    
    # Now let's index all documents together
    print_colored("\nCreating unified index of all documents...", Colors.YELLOW)
    
    # Define the index path to check if it exists
    index_path = os.path.join(os.path.dirname(current_file), ".byaldi", unified_index_name)
    
    # Check if unified index already exists
    if os.path.exists(index_path):
        print_colored("Unified index already exists. Attempting to load...", Colors.GREEN)
        try:
            unified_rag_model.load_index(unified_index_name)
            print_colored("Successfully loaded existing unified index!", Colors.GREEN)
        except Exception as e:
            print_colored(f"Error loading existing index: {e}", Colors.RED)
            print_colored("Will create a new unified index...", Colors.YELLOW)
            # We'll use a custom indexing method since we want to combine multiple documents
            create_unified_index(unified_rag_model, unified_index_name, all_pdf_images)
    else:
        # Create a new unified index
        create_unified_index(unified_rag_model, unified_index_name, all_pdf_images)
    
    # Start interactive mode with the unified model
    print_colored("\n✨ All documents processed successfully into a unified index! Starting interactive mode...", Colors.GREEN)
    interactive_unified_mode(unified_rag_model, all_pdf_images, pdf_page_mapping, api_key=api_key)

if __name__ == "__main__":
    main()