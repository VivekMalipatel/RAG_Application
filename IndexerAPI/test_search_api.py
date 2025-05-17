import requests
import json
import sys
import os

def test_search_text_api(text, top_k=10):
    """
    Test the /search/text API endpoint with the provided query text
    
    Args:
        text (str): The text query to search for
        top_k (int): Number of results to return (default: 10)
    
    Returns:
        dict: The API response
    """
    # API endpoint URL - update with the correct host and port if needed
    url = "http://localhost:8009/vector/search/text"
    
    # Prepare request payload
    payload = {
        "text": text,
        "top_k": top_k
    }
    
    # Send POST request to API
    try:
        response = requests.post(url, json=payload)
        
        # Check if request was successful
        if response.status_code == 200:
            results = response.json()
            return results
        else:
            print(f"Error: API returned status code {response.status_code}")
            print(f"Response: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return None

def ingest_file(file_path, source, metadata=None):
    """
    Ingest a file by calling the Ingest/file API endpoint
    
    Args:
        file_path (str): Path to the file to ingest
        source (str): Source identifier for the file
        metadata (dict, optional): Additional metadata for the file
    
    Returns:
        dict or None: The API response or None if the request failed
    """
    # API endpoint URL - update with the correct host and port if needed
    url = "http://localhost:8009/ingest/file"
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return None
    
    # Prepare form data
    form_data = {
        'source': source
    }
    
    # Add metadata if provided
    if metadata:
        form_data['metadata'] = json.dumps(metadata)
    
    # Prepare the file
    filename = os.path.basename(file_path)
    files = {
        'file': (filename, open(file_path, 'rb'))
    }
    
    # Send POST request to API
    try:
        response = requests.post(url, data=form_data, files=files)
        
        # Close file handler
        files['file'][1].close()
        
        # Check if request was successful
        if response.status_code == 200:
            result = response.json()
            print(f"Successfully ingested file: {filename}")
            print(f"Queue ID: {result.get('id')}")
            print(f"Message: {result.get('message')}")
            return result
        else:
            print(f"Error: API returned status code {response.status_code}")
            print(f"Response: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        # Make sure to close file handler in case of error
        files['file'][1].close()
        print(f"Request error: {e}")
        return None

def display_results(results):
    """
    Display search results in a formatted way
    
    Args:
        results (list): List of search results
    """
    if not results:
        print("No results to display")
        return
    
    print(f"\nFound {len(results)} results:")
    print("-" * 60)
    
    for i, result in enumerate(results, 1):
        print(f"Result #{i}:")
        print(f"Score: {result.get('score', 'N/A')}")
        print(f"Text: {result.get('text', 'N/A')[:100]}...")  # Show first 100 chars
        
        # Display metadata if available
        metadata = result.get('metadata', {})
        if metadata:
            print("Metadata:")
            for key, value in metadata.items():
                print(f"  - {key}: {value}")
        
        print("-" * 60)

def ingest_files_from_directory(directory_path, source, extensions=None):
    """
    Ingest all files from a directory by calling the Ingest/file API endpoint
    
    Args:
        directory_path (str): Path to the directory containing files to ingest
        source (str): Source identifier for the files
        extensions (list, optional): List of file extensions to include (e.g., ['.pdf', '.docx'])
                                    If None, all files will be ingested
    
    Returns:
        list: List of queue IDs for the ingested files
    """
    if not os.path.isdir(directory_path):
        print(f"Error: Directory not found at {directory_path}")
        return []
    
    queue_ids = []
    
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        
        # Skip directories
        if os.path.isdir(file_path):
            continue
        
        # Check file extension if specified
        if extensions:
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext not in extensions:
                continue
        
        # Create metadata with filename
        metadata = {
            'filename': filename,
            'original_path': file_path
        }
        
        # Ingest the file
        result = ingest_file(file_path, source, metadata)
        
        if result and 'id' in result:
            queue_ids.append(result['id'])
    
    print(f"Ingested {len(queue_ids)} files from {directory_path}")
    return queue_ids

if __name__ == "__main__":
    # Check for command argument to determine action
    if len(sys.argv) > 1 and sys.argv[1] == "ingest":
        # File ingestion mode
        if len(sys.argv) < 3:
            print("Usage: python test_search_api.py ingest <file_path> [source]")
            print("   or: python test_search_api.py ingest-dir <directory_path> [source] [extension1,extension2,...]")
            sys.exit(1)
        
        if sys.argv[1] == "ingest":
            file_path = sys.argv[2]
            source = sys.argv[3] if len(sys.argv) > 3 else "test-import"
            
            # Optional metadata
            metadata = None
            if len(sys.argv) > 4:
                try:
                    metadata = json.loads(sys.argv[4])
                except json.JSONDecodeError:
                    print("Warning: Invalid JSON metadata format. Using empty metadata.")
            
            ingest_file(file_path, source, metadata)
        
        elif sys.argv[1] == "ingest-dir":
            dir_path = sys.argv[2]
            source = sys.argv[3] if len(sys.argv) > 3 else "test-import"
            
            # Optional extensions filter
            extensions = None
            if len(sys.argv) > 4:
                extensions = [f".{ext.lower().strip('.')}" for ext in sys.argv[4].split(',')]
            
            ingest_files_from_directory(dir_path, source, extensions)
    
    else:
        # Interactive search mode
        print("=== Interactive Search Mode ===")
        print("Type 'exit' or 'quit' to end the program")
        
        while True:
            # Get search query from user input
            query = input("\nEnter your search query: ")
            
            # Check if user wants to exit
            if query.lower() in ['exit', 'quit']:
                print("Exiting search mode...")
                break
            
            # Skip empty queries
            if not query.strip():
                print("Please enter a valid query")
                continue
            
            # Get number of results
            try:
                k_input = input("Number of results to return (default: 10): ")
                k = 10 if not k_input.strip() else int(k_input)
            except ValueError:
                print("Invalid number, using default (10)")
                k = 10
            
            print(f"\nSearching for: '{query}' (top {k} results)")
            
            # Run the API test
            results = test_search_text_api(query, k)
            
            # Display results if available
            if results:
                display_results(results)
