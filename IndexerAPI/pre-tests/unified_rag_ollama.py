#!/usr/bin/env python
"""
Unified RAG Over PDFs with ColNomic Embed Multimodal and Ollama Vision API

This script implements a multimodal retrieval-augmented generation system
for querying multiple PDF documents using ColNomic Embed for retrieval
and a local Ollama Vision model for generation. It provides a unified
conversational interface across all indexed documents.
"""

# Set tokenizers parallelism to False to avoid warnings
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import base64
import io
import json
import sys
import logging
from pathlib import Path
import tempfile
import time  # Added time import for sleep functionality

import matplotlib.pyplot as plt
from pdf2image import convert_from_path
from PIL import Image
import requests
import torch
from typing import Dict, List, Optional

# Attempt to import necessary libraries
try:
    from byaldi import RAGMultiModalModel
except ImportError:
    print("Error: 'byaldi' library not found. Please install it (`pip install byaldi`).")
    sys.exit(1)

try:
    from markitdown import MarkItDown
except ImportError:
    print("Error: 'markitdown' library not found. Please install it (`pip install markitdown`).")
    # Consider exiting if MarkItDown is crucial, or proceed without Markdown conversion
    # sys.exit(1)
    MarkItDown = None # Allow script to run without MarkItDown for now

# --- Configuration ---

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set up custom directory for model storage (absolute path)
current_file = os.path.abspath(__file__)
models_dir = os.path.join(os.path.dirname(current_file), ".models")
index_dir = os.path.join(os.path.dirname(current_file), ".byaldi") # Directory for ByAldi indices
markdown_dir = os.path.join(os.path.dirname(current_file), "markdown_output") # Directory for Markdown output

# Create directories if they don't exist
os.makedirs(models_dir, exist_ok=True)
os.makedirs(index_dir, exist_ok=True)
os.makedirs(markdown_dir, exist_ok=True)

# Set environment variables to force HF to use our custom cache directory
os.environ["HF_HOME"] = models_dir
os.environ["TRANSFORMERS_CACHE"] = os.path.join(models_dir, "transformers")
os.environ["HF_DATASETS_CACHE"] = os.path.join(models_dir, "datasets")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(models_dir, "hub")

# Print the cache locations to verify they're set correctly
print(f"HF_HOME set to: {os.environ['HF_HOME']}")
print(f"TRANSFORMERS_CACHE set to: {os.environ['TRANSFORMERS_CACHE']}")

# Device Configuration
device = "cuda:0" if torch.cuda.is_available() else "mps" # Changed default to CPU
logger.info(f"Using device: {device}")
logger.info(f"Models will be stored at: {models_dir}")
logger.info(f"Indices will be stored at: {index_dir}")
logger.info(f"Markdown output will be stored at: {markdown_dir}")

# Define paths to PDFs (relative to the script directory)
script_dir = os.path.dirname(current_file)
PDF_FILES = {
    "Resume": Path(os.path.join(script_dir, "Resume.pdf")),
    "RAGAS_Paper": Path(os.path.join(script_dir, "ragas_papers.pdf")),
    # Add more PDFs here if needed
}

# Ollama Configuration
OLLAMA_API_BASE = os.environ.get("OLLAMA_API_BASE", "http://10.9.0.6:11434/api") # Default to localhost
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2-vision:11b-instruct-q8_0") # Or your preferred model
logger.info(f"Using Ollama server at: {OLLAMA_API_BASE}")
logger.info(f"Using Ollama model: {OLLAMA_MODEL}")

# RAG Configuration
UNIFIED_INDEX_NAME = "unified_pdf_index_v1"
RETRIEVAL_K = 2 # Number of pages to retrieve

# ColNomic/ByAldi Model Configuration
RETRIEVAL_MODEL_NAME = "nomic-ai/colnomic-embed-multimodal-3b"

# --- End Configuration ---


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

def display_pdf_images(images_list, title="PDF Preview", max_preview=5):
    """Display the first few images in the provided list."""
    num_images_to_show = min(len(images_list), max_preview)
    if (num_images_to_show == 0):
        return

    num_cols = min(5, num_images_to_show)
    num_rows = (num_images_to_show + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows))
    if (num_rows == 1 and num_cols == 1):
        axes = [axes] # Make it iterable
    else:
        axes = axes.flatten()

    for i in range(num_images_to_show):
        ax = axes[i]
        ax.imshow(images_list[i])
        ax.set_title(f"{title} - Page {i+1}")
        ax.axis('off')

    # Hide empty subplots
    for j in range(num_images_to_show, len(axes)):
        axes[j].axis('off')

    plt.suptitle(f"{title} (First {num_images_to_show} Pages)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    plt.show()

def convert_pdf_to_images(pdf_path: Path) -> List[Image.Image]:
    """Convert PDF pages to PIL Images."""
    try:
        images = convert_from_path(str(pdf_path))
        logger.info(f"Successfully converted {pdf_path.name} to {len(images)} images.")
        return images
    except Exception as e:
        logger.error(f"Error converting {pdf_path.name} to images: {e}")
        return []

def convert_pdf_to_markdown(pdf_path: Path, output_dir: str) -> Optional[Path]:
    """Convert a PDF file to markdown using MarkItDown and save it."""
    if MarkItDown is None:
        logger.warning("MarkItDown library not available. Skipping Markdown conversion.")
        return None

    logger.info(f"Converting {pdf_path.name} to Markdown...")
    md_converter = MarkItDown(enable_plugins=True) # Enable plugins for PDF
    # Convert output_dir to Path object if it's a string
    output_dir = Path(output_dir) if isinstance(output_dir, str) else output_dir
    output_md_path = output_dir / f"{pdf_path.stem}.md"

    try:
        result = md_converter.convert(str(pdf_path))
        with open(output_md_path, 'w', encoding='utf-8') as f:
            f.write(result.text_content)
        logger.info(f"Successfully converted {pdf_path.name} and saved Markdown to {output_md_path}")
        return output_md_path
    except Exception as e:
        logger.error(f"Error converting {pdf_path.name} to Markdown: {e}")
        return None

def create_rag_model() -> Optional[RAGMultiModalModel]:
    """Initialize the RAG model with ColNomic Embed Multimodal."""
    # Check if we have an existing index
    index_path = Path(index_dir) / UNIFIED_INDEX_NAME
    
    if index_path.exists():
        print_colored(f"Found existing index at {index_path}. Loading model from index...", Colors.YELLOW)
        try:
            # Use the from_index class method to load the model with existing index
            rag = RAGMultiModalModel.from_index(
                index_path=index_path,
                index_root=index_dir,
                device=device,
                verbose=1
            )
            print_colored("Model loaded successfully from existing index!", Colors.GREEN)
            return rag
        except Exception as e:
            print_colored(f"Error loading model from index: {e}", Colors.RED)
            print_colored("Will try loading model from pretrained checkpoint instead.", Colors.YELLOW)
            # Fall through to loading from pretrained
    
    # If no index or loading from index failed, load from pretrained checkpoint
    print_colored(f"Loading RAG model: {RETRIEVAL_MODEL_NAME}...", Colors.YELLOW)
    try:
        # Initialize using device and cache directory
        rag = RAGMultiModalModel.from_pretrained(
            RETRIEVAL_MODEL_NAME,
            index_root=index_dir,
            device=device,
            verbose=1
        )
        print_colored("RAG model loaded successfully!", Colors.GREEN)
        return rag
    except Exception as e:
        print_colored(f"Error loading RAG model: {e}", Colors.RED)
        logger.exception("Detailed error during RAG model loading:")
        return None

def encode_image_to_base64(image: Image.Image) -> str:
    """Encode PIL image to base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG") # Use JPEG for potentially smaller size
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def query_ollama_vision(query: str, images: List[Image.Image], conversation_history: List[Dict]) -> Optional[str]:
    """
    Query Ollama Vision model with text and images (non-streaming).
    Updates conversation_history in place.
    """
    base64_images = [encode_image_to_base64(img) for img in images if img]
    if not base64_images:
        logger.error("No valid images to send to Ollama.")
        return "Error: Could not process images for the query."

    # Prepare the user message
    user_message = {
        "role": "user",
        "content": query,
        "images": base64_images
    }

    # Ensure history starts with system prompt if empty
    if not conversation_history:
        conversation_history.append({
            "role": "system",
            "content": "You are an expert PDF analyst. Analyze the provided PDF page images and answer questions thoroughly based solely on the visible content. Be concise but comprehensive. If the answer isn't in the provided pages, state that clearly."
        })

    messages_for_api = conversation_history + [user_message]

    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages_for_api,
        "stream": False
    }

    try:
        logger.info(f"Sending request to Ollama ({OLLAMA_MODEL}) with query and {len(base64_images)} image(s)...")
        response = requests.post(
            f"{OLLAMA_API_BASE}/chat",
            json=payload,
            timeout=180 # Increased timeout
        )
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        response_data = response.json()
        assistant_message = response_data.get("message", {})
        content = assistant_message.get("content")

        if content:
            # Add user query and assistant response to the main history
            conversation_history.append(user_message)
            conversation_history.append(assistant_message)
            logger.info("Received response from Ollama.")
            return content
        else:
            logger.error(f"Ollama response missing content: {response_data}")
            # Add user query but indicate error in history
            conversation_history.append(user_message)
            conversation_history.append({"role": "assistant", "content": "Error: Received an empty response from the model."})
            return "Error: Received an empty response from the model."

    except requests.exceptions.Timeout:
        logger.error("Ollama request timed out.")
        conversation_history.append(user_message)
        conversation_history.append({"role": "assistant", "content": "Error: Request to Ollama timed out."})
        return "Error: Request to Ollama timed out."
    except requests.exceptions.RequestException as e:
        logger.error(f"Ollama network error: {e}")
        conversation_history.append(user_message)
        conversation_history.append({"role": "assistant", "content": f"Error: Network error connecting to Ollama: {e}"})
        return f"Error: Network error connecting to Ollama: {e}"
    except Exception as e:
        logger.error(f"Error during Ollama query: {e}")
        logger.exception("Detailed error during Ollama query:")
        conversation_history.append(user_message)
        conversation_history.append({"role": "assistant", "content": f"Error: An unexpected error occurred: {e}"})
        return f"Error: An unexpected error occurred: {e}"

def query_ollama_vision_streaming(query: str, images: List[Image.Image], conversation_history: List[Dict]) -> Optional[str]:
    """
    Query Ollama Vision model with text and images (streaming).
    Updates conversation_history in place.
    Returns the complete response text.
    """
    base64_images = [encode_image_to_base64(img) for img in images if img]
    if not base64_images:
        logger.error("No valid images to send to Ollama.")
        return "Error: Could not process images for the query."

    # Prepare the user message
    user_message = {
        "role": "user",
        "content": query,
        "images": base64_images
    }

    # Ensure history starts with system prompt if empty
    if not conversation_history:
        conversation_history.append({
            "role": "system",
            "content": "You are an expert PDF analyst. Analyze the provided PDF page images and answer questions thoroughly based solely on the visible content. Be concise but comprehensive. If the answer isn't in the provided pages, state that clearly."
        })

    messages_for_api = conversation_history + [user_message]

    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages_for_api,
        "stream": True  # Enable streaming
    }

    # Retry parameters
    max_retries = 3
    retry_count = 0
    success = False

    while retry_count < max_retries and not success:
        try:
            if retry_count > 0:
                print_colored(f"\nRetrying request (attempt {retry_count+1}/{max_retries})...", Colors.YELLOW)
            
            logger.info(f"Sending streaming request to Ollama ({OLLAMA_MODEL}) with query and {len(base64_images)} image(s)...")
            
            # Increased timeout for vision models which can take longer
            timeout = 300 # 5 minutes timeout
            
            response = requests.post(
                f"{OLLAMA_API_BASE}/chat",
                json=payload,
                stream=True,  # Request streaming response
                timeout=timeout
            )
            
            # Check if we got an error response
            if response.status_code != 200:
                error_msg = f"Ollama returned status code {response.status_code}"
                try:
                    # Try to extract error details from response
                    error_data = response.json()
                    if "error" in error_data:
                        error_msg += f": {error_data['error']}"
                except:
                    # If we can't parse JSON, just use the text
                    error_msg += f": {response.text}"
                
                logger.error(error_msg)
                retry_count += 1
                
                if retry_count < max_retries:
                    # If we have retries left, try again with fewer images or fallback to non-streaming
                    if len(base64_images) > 1 and retry_count == 1:
                        print_colored("Reducing image count to improve reliability...", Colors.YELLOW)
                        # Keep only the first image to reduce payload size
                        base64_images = [base64_images[0]]
                        user_message["images"] = base64_images
                        messages_for_api = conversation_history + [user_message]
                        payload["messages"] = messages_for_api
                    elif retry_count == 2:
                        # On second retry, switch to non-streaming mode
                        print_colored("Switching to non-streaming mode...", Colors.YELLOW)
                        payload["stream"] = False
                    
                    # Add a short delay before retrying
                    time.sleep(2)
                    continue
                else:
                    # We've exhausted our retries
                    conversation_history.append(user_message)
                    conversation_history.append({"role": "assistant", "content": f"Error: {error_msg} after {max_retries} attempts."})
                    return f"Error: {error_msg} after {max_retries} attempts."
            
            # For collecting complete response
            full_response = ""
            assistant_message = {"role": "assistant", "content": ""}
            
            print("")  # Start on a new line
            
            # Process the streaming response
            for line in response.iter_lines():
                if line:
                    try:
                        # Decode line and parse JSON
                        chunk_data = json.loads(line.decode('utf-8'))
                        
                        # Extract the text chunk
                        if 'message' in chunk_data and 'content' in chunk_data['message']:
                            chunk = chunk_data['message']['content']
                            full_response += chunk
                            assistant_message["content"] += chunk
                            
                            # Print chunk immediately without newline
                            print(chunk, end='', flush=True)
                        
                        # Check if this is the final message
                        if chunk_data.get('done', False):
                            print("")  # Add a newline at the end
                            success = True
                            break
                            
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to decode JSON from stream: {line}")
            
            # If we got here without setting success to True, something went wrong
            if not success and retry_count < max_retries - 1:
                logger.warning("Stream ended unexpectedly, retrying...")
                retry_count += 1
                time.sleep(2)
                continue
                
            # Add user query and complete assistant response to conversation history
            conversation_history.append(user_message)
            conversation_history.append(assistant_message)
            logger.info("Completed streaming response from Ollama.")
            return full_response
            
        except requests.exceptions.Timeout:
            logger.error("Ollama streaming request timed out.")
            retry_count += 1
            
            if retry_count < max_retries:
                # If we have retries left, try again with simplified parameters
                if retry_count == 1:
                    # Reduce the number of images on first timeout
                    if len(base64_images) > 1:
                        print_colored("\nTimeout occurred. Reducing image count and retrying...", Colors.YELLOW)
                        base64_images = [base64_images[0]] # Just keep the first image
                        user_message["images"] = base64_images
                        messages_for_api = conversation_history + [user_message]
                        payload["messages"] = messages_for_api
                elif retry_count == 2:
                    # Switch to non-streaming mode on second timeout
                    print_colored("\nTimeout occurred again. Switching to non-streaming mode...", Colors.YELLOW)
                    payload["stream"] = False
                
                time.sleep(2) # Wait before retrying
                continue
            else:
                conversation_history.append(user_message)
                conversation_history.append({"role": "assistant", "content": f"Error: Request to Ollama timed out after {max_retries} attempts."})
                return f"Error: Request to Ollama timed out after {max_retries} attempts."
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama network error: {e}")
            retry_count += 1
            
            if retry_count < max_retries:
                print_colored(f"\nNetwork error: {e}. Retrying...", Colors.RED)
                time.sleep(2)
                continue
            else:
                conversation_history.append(user_message)
                conversation_history.append({"role": "assistant", "content": f"Error: Network error connecting to Ollama: {e}"})
                return f"Error: Network error connecting to Ollama: {e}"
        except Exception as e:
            logger.error(f"Error during Ollama streaming query: {e}")
            logger.exception("Detailed error during Ollama streaming query:")
            
            retry_count += 1
            if retry_count < max_retries:
                print_colored(f"\nUnexpected error: {e}. Retrying...", Colors.RED)
                time.sleep(2)
                continue
            else:
                conversation_history.append(user_message)
                conversation_history.append({"role": "assistant", "content": f"Error: An unexpected error occurred: {e}"})
                return f"Error: An unexpected error occurred: {e}"

def build_unified_index(rag_model: RAGMultiModalModel, pdf_files: Dict[str, Path], index_name: str, force_reindex: bool = False):
    """
    Builds or loads a unified index for multiple PDF files using ByAldi.
    ByAldi's index function handles PDF conversion internally.
    """
    # Convert index_dir to Path if it's a string
    index_dir_path = Path(index_dir) if isinstance(index_dir, str) else index_dir
    index_path = index_dir_path / index_name
    
    # Check if index exists and if we should load it
    if index_path.exists() and not force_reindex:
        print_colored(f"Unified index '{index_name}' already exists. Loading...", Colors.GREEN)
        try:
            # The proper way to load an existing index is to use from_index
            # But we already have a model instance, so we'll trust the automatic loading
            print_colored(f"Index '{index_name}' is ready.", Colors.GREEN)
            return True
        except Exception as e:
            print_colored(f"Error loading existing index '{index_name}': {e}. Will re-index.", Colors.RED)
            force_reindex = True
    
    # If we need to create a new index or re-index
    if not index_path.exists() or force_reindex:
        if force_reindex and index_path.exists():
            print_colored(f"Forcing re-indexing for '{index_name}'.", Colors.YELLOW)
            # We'll let the overwrite parameter handle this

        print_colored(f"Building unified index '{index_name}'...", Colors.YELLOW)
        all_pdf_paths = []
        valid_pdf_names = []
        for name, pdf_path in pdf_files.items():
            if pdf_path.exists():
                all_pdf_paths.append(str(pdf_path))
                valid_pdf_names.append(name)
            else:
                print_colored(f"Warning: PDF file not found: {pdf_path}. Skipping.", Colors.RED)

        if not all_pdf_paths:
            print_colored("Error: No valid PDF files found to index.", Colors.RED)
            return False

        print_colored(f"Indexing the following documents: {', '.join(valid_pdf_names)}", Colors.BLUE)

        try:
            # Process the first PDF to create the initial index
            first_pdf_path = all_pdf_paths[0]
            print_colored(f"Indexing first file: {first_pdf_path}", Colors.YELLOW)
            rag_model.index(
                input_path=first_pdf_path,
                index_name=index_name,
                store_collection_with_index=True,  # Store with index to ensure embeddings are available
                overwrite=True  # Overwrite existing index if needed
            )
            
            # Process additional PDFs by adding them to the index
            if len(all_pdf_paths) > 1:
                for i, pdf_path in enumerate(all_pdf_paths[1:], start=1):
                    print_colored(f"Adding to index: {pdf_path}", Colors.YELLOW)
                    rag_model.add_to_index(
                        input_item=pdf_path,
                        store_collection_with_index=True,
                        doc_id=i,
                        metadata={"document_name": valid_pdf_names[i]}
                    )
            
            print_colored(f"Unified index '{index_name}' built successfully!", Colors.GREEN)
            return True
        except Exception as e:
            print_colored(f"Error building unified index: {e}", Colors.RED)
            logger.exception("Detailed error during index building:")
            return False
    
    return False

def interactive_unified_mode(rag_model: RAGMultiModalModel, all_images_map: Dict[str, List[Image.Image]], page_mapping: Dict[int, Dict]):
    """
    Run the RAG system in interactive conversation mode with unified document pool using Ollama.
    """
    conversation_history = [] # Unified conversation history
    use_streaming = True # Set to True to use streaming, False for non-streaming

    print_colored("\n" + "="*80, Colors.BOLD)
    print_colored(" 🤖 Unified PDF RAG System with Ollama Vision 📑", Colors.BOLD)
    print_colored("="*80, Colors.BOLD)

    docs_info = {}
    for info in page_mapping.values():
        doc_name = info["document"]
        if doc_name not in docs_info:
            docs_info[doc_name] = 0
        docs_info[doc_name] += 1

    print_colored(f"\nIndexed {len(docs_info)} documents with {len(page_mapping)} total pages:", Colors.GREEN)
    for doc, count in docs_info.items():
        print_colored(f"  • {doc}: {count} pages", Colors.BLUE)

    print_colored("\nCommands:", Colors.YELLOW)
    print_colored("  !help       - Show this help message", Colors.BLUE)
    print_colored("  !clear      - Clear conversation history", Colors.BLUE)
    print_colored("  !stream     - Toggle streaming mode (currently " + ("ON" if use_streaming else "OFF") + ")", Colors.BLUE)
    print_colored("  !quit/!exit - Exit the program", Colors.BLUE)
    print_colored("\nEnter your query about the documents.", Colors.YELLOW)

    while True:
        try:
            print_colored("\n> ", Colors.BOLD, end='')
            user_input = input().strip()

            if not user_input:
                continue

            if user_input.lower() in ["!quit", "!exit", "quit", "exit"]:
                print_colored("\nGoodbye!", Colors.GREEN)
                break

            elif user_input.lower() == "!help":
                print_colored("\nCommands:", Colors.YELLOW)
                print_colored("  !help       - Show this help message", Colors.BLUE)
                print_colored("  !clear      - Clear conversation history", Colors.BLUE)
                print_colored("  !stream     - Toggle streaming mode (currently " + ("ON" if use_streaming else "OFF") + ")", Colors.BLUE)
                print_colored("  !quit/!exit - Exit the program", Colors.BLUE)
                print_colored("\nIndexed Documents:", Colors.GREEN)
                for doc, count in docs_info.items():
                    print_colored(f"  • {doc}: {count} pages", Colors.BLUE)
                continue

            elif user_input.lower() == "!clear":
                conversation_history = []
                print_colored("\nConversation history cleared.", Colors.YELLOW)
                continue
                
            elif user_input.lower() == "!stream":
                use_streaming = not use_streaming
                print_colored(f"\nStreaming mode is now: {Colors.GREEN if use_streaming else Colors.RED}{('ON' if use_streaming else 'OFF')}{Colors.ENDC}", Colors.BLUE)
                continue

            # Process the query
            print_colored(f"\nSearching across documents for: '{user_input}'...", Colors.YELLOW)

            # Search relevant pages using the RAG model
            try:
                # Try different versions of the search API
                try:
                    # First try with just query and k
                    rag_results = rag_model.search(user_input, k=RETRIEVAL_K)
                except TypeError:
                    # If that fails, try with just the query
                    try:
                        rag_results = rag_model.search(user_input)
                    except Exception as e:
                        # Last resort: try model.search if available
                        if hasattr(rag_model, 'model') and hasattr(rag_model.model, 'search'):
                            rag_results = rag_model.model.search(user_input, k=RETRIEVAL_K)
                        else:
                            raise e
            except Exception as e:
                 print_colored(f"Error during search: {e}", Colors.RED)
                 logger.exception("Detailed error during search:")
                 continue # Skip to next input

            if not rag_results:
                print_colored("No relevant pages found for your query.", Colors.RED)
                conversation_history.append({"role": "user", "content": user_input})
                conversation_history.append({"role": "assistant", "content": "I couldn't find any relevant pages across the documents for your query."})
                continue

            # Debug print to see the actual structure of rag_results
            logger.debug(f"Search results type: {type(rag_results)}")
            if rag_results and len(rag_results) > 0:
                logger.debug(f"First result type: {type(rag_results[0])}")
                logger.debug(f"First result attributes: {dir(rag_results[0])}")

            # Retrieve images and map back to original document/page
            retrieved_images = []
            source_info_list = []
            retrieved_page_details = [] # Store details for the prompt

            for result in rag_results:
                # Handle different types of result objects from ByAldi
                # Check if result is a dict
                if isinstance(result, dict):
                    unified_page_idx_1based = result.get("page_num")
                    doc_name_from_result = result.get("doc_name")
                # Check if result is an object with attributes
                elif hasattr(result, "page_num"):
                    unified_page_idx_1based = result.page_num
                    doc_name_from_result = getattr(result, "doc_name", None)
                # For ByAldi's Result object structure
                else:
                    # Try to infer the page number from what we have
                    # Assuming the first item in the result might be the page number
                    try:
                        # Check if result is indexable
                        if hasattr(result, "__getitem__"):
                            # Try different fields that might contain page info
                            if "page_num" in result:
                                unified_page_idx_1based = result["page_num"]
                            elif "document" in result and "page" in result:
                                # If result has document and page fields
                                doc_name_from_result = result["document"]
                                unified_page_idx_1based = result["page"] + 1
                            else:
                                # As a last resort, try to use the first element
                                unified_page_idx_1based = result[0] if len(result) > 0 else None
                                doc_name_from_result = None
                        else:
                            # If we can't index the result, convert to string and try to parse
                            result_str = str(result)
                            print_colored(f"DEBUG: Result as string: {result_str}", Colors.YELLOW)
                            
                            # Try to extract page_num if visible in the string representation
                            import re
                            page_match = re.search(r'page[_\s]*(?:num|number)[:\s=]*(\d+)', result_str, re.I)
                            if page_match:
                                unified_page_idx_1based = int(page_match.group(1))
                            else:
                                # Last attempt: if it's a numeric value (index)
                                unified_page_idx_1based = int(float(result)) if str(result).replace('.', '', 1).isdigit() else None
                            doc_name_from_result = None
                    except Exception as extract_err:
                        logger.warning(f"Failed to extract page number from search result: {extract_err}")
                        print_colored(f"DEBUG: Result type: {type(result)}, dir: {dir(result)}", Colors.YELLOW)
                        unified_page_idx_1based = None
                        doc_name_from_result = None

                if unified_page_idx_1based is None:
                    logger.warning(f"Skipping result with missing page number: {result}")
                    continue

                unified_page_idx_0based = unified_page_idx_1based - 1

                if unified_page_idx_0based in page_mapping:
                    page_info = page_mapping[unified_page_idx_0based]
                    doc_name = page_info["document"]
                    original_page_idx_0based = page_info["original_page"]
                    display_page = page_info["display_page"]

                    # Get the image from the pre-loaded map
                    if doc_name in all_images_map and 0 <= original_page_idx_0based < len(all_images_map[doc_name]):
                        img = all_images_map[doc_name][original_page_idx_0based]
                        retrieved_images.append(img)
                        source_desc = f"{doc_name} (Page {display_page})"
                        source_info_list.append(source_desc)
                        retrieved_page_details.append(f"Source: {source_desc}, Unified Index Page: {unified_page_idx_1based}")
                    else:
                        logger.warning(f"Could not find image for {doc_name} page {display_page} (Unified index {unified_page_idx_1based})")
                else:
                    # Fallback if mapping fails (should not happen if indexing matches mapping)
                    logger.warning(f"Could not map unified page index {unified_page_idx_0based} back to original document.")
                    # Try using doc_name from result if available
                    if doc_name_from_result and doc_name_from_result in all_images_map:
                         # We don't know the original page number easily here without more info from ByAldi
                         # Let's try to find *any* image from that doc as a fallback (not ideal)
                         if all_images_map[doc_name_from_result]:
                              retrieved_images.append(all_images_map[doc_name_from_result][0]) # Just take the first page
                              source_desc = f"{doc_name_from_result} (Page Unknown, Unified Index Page: {unified_page_idx_1based})"
                              source_info_list.append(source_desc)
                              retrieved_page_details.append(f"Source: {source_desc}")

            if not retrieved_images:
                print_colored("Found relevant page numbers, but failed to retrieve corresponding images.", Colors.RED)
                continue

            print_colored("Found relevant content from:", Colors.GREEN)
            for source_desc in source_info_list:
                print_colored(f"  • {source_desc}", Colors.BLUE)

            # Generate response using Ollama
            print_colored("\nGenerating response using Ollama", Colors.BOLD)
            if use_streaming:
                print_colored(" (streaming):", Colors.BOLD)
            else:
                print_colored(":", Colors.BOLD)
            sys.stdout.flush()

            # Prepare context query for Ollama
            context_query = (
                f"Based *only* on the content visible in the following provided page images, "
                f"please answer the user's question: '{user_input}'\n\n"
                f"The images correspond to these sources: [{'; '.join(source_info_list)}]."
                # f"Details: [{'; '.join(retrieved_page_details)}]" # Optional: Add more detail if needed
            )

            # Call Ollama with either streaming or non-streaming based on user preference
            if use_streaming:
                response_text = query_ollama_vision_streaming(context_query, retrieved_images, conversation_history)
            else:
                response_text = query_ollama_vision(context_query, retrieved_images, conversation_history)
                if response_text:
                    print(response_text)  # Print the full response for non-streaming mode

            print("\n") # Add newline after response

        except KeyboardInterrupt:
            print_colored("\n\nInterrupted. Enter !quit to exit or continue.", Colors.YELLOW)
        except Exception as e:
            print_colored(f"\nAn error occurred in the interactive loop: {e}", Colors.RED)
            logger.exception("Exception in interactive loop:")

def main():
    """Main function to initialize models, process PDFs, build index, and start interactive mode."""
    print_colored("\n🚀 Starting Unified Multimodal RAG System with Ollama Vision...", Colors.BOLD)
    # 1. Check Ollama Connection
    try:
        response = requests.head(OLLAMA_API_BASE.replace("/api", ""), timeout=5) # Check base URL
        response.raise_for_status()
        print_colored(f"Successfully connected to Ollama server at {OLLAMA_API_BASE}", Colors.GREEN)
    except requests.exceptions.RequestException as e:
        print_colored(f"Error: Could not connect to Ollama server at {OLLAMA_API_BASE}.", Colors.RED)
        print_colored(f"Details: {e}", Colors.RED)
        print_colored("Please ensure Ollama is running and accessible.", Colors.YELLOW)
        # Decide if you want to exit or continue without Ollama
        # return # Exit if Ollama is essential
        print_colored("Proceeding without guaranteed Ollama connection...", Colors.YELLOW)


    # 2. Initialize RAG Model
    rag_model = create_rag_model()
    if rag_model is None:
        print_colored("Failed to initialize RAG model. Exiting.", Colors.RED)
        return

    # 3. Process PDFs: Convert to Images and optionally Markdown
    all_images_map = {} # Stores images: {"DocName": [img1, img2,...]}
    page_mapping = {}   # Stores mapping: {unified_idx_0based: {"document": name, "original_page": idx, "display_page": idx+1}}
    current_unified_page_index = 0

    print_colored("\nProcessing PDF documents...", Colors.YELLOW)
    valid_pdfs_for_indexing = {}
    for name, pdf_path in PDF_FILES.items():
        if not pdf_path.exists():
            print_colored(f"Warning: PDF file not found: {pdf_path}. Skipping.", Colors.RED)
            continue

        # Convert PDF to images
        images = convert_pdf_to_images(pdf_path)
        if not images:
            print_colored(f"Failed to convert {pdf_path.name} to images. Skipping.", Colors.RED)
            continue

        all_images_map[name] = images
        valid_pdfs_for_indexing[name] = pdf_path # Keep track of valid PDFs for indexing

        # Create page mapping for this document
        for i, _ in enumerate(images):
            page_mapping[current_unified_page_index] = {
                "document": name,
                "original_page": i,          # 0-based index within the original PDF
                "display_page": i + 1        # 1-based index for display
            }
            current_unified_page_index += 1

        # Convert to Markdown (optional)
        convert_pdf_to_markdown(pdf_path, markdown_dir)

    if not all_images_map:
        print_colored("Error: No PDF documents were successfully processed. Exiting.", Colors.RED)
        return

    # 4. Build or Load Unified RAG Index
    # Pass only the valid PDFs found
    index_built = build_unified_index(rag_model, valid_pdfs_for_indexing, UNIFIED_INDEX_NAME)
    if not index_built:
         print_colored("Failed to build or load the unified index. Cannot proceed with querying.", Colors.RED)
         # Decide if you want to exit or allow interaction without a working index
         # return # Exit if index is essential

    # 5. Start Interactive Mode
    print_colored("\n✨ Setup complete! Starting interactive mode...", Colors.GREEN)
    interactive_unified_mode(rag_model, all_images_map, page_mapping)

if __name__ == "__main__":
    main()
