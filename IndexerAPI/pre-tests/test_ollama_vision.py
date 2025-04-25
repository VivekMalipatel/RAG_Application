#!/usr/bin/env python
"""
Test script for Ollama Vision models using the Ollama API.
Sends local images to an Ollama vision model and gets back descriptions.
"""

import os
import sys
import json
import base64
import requests
import argparse
from pathlib import Path
import logging
from PIL import Image

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
OLLAMA_API_BASE = "http://10.9.0.6:11434/api"
DEFAULT_MODEL = "llama3.2-vision:11b-instruct-q8_0"

def encode_image_to_base64(image_path):
    """Convert an image file to base64 encoding"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Failed to encode image {image_path}: {e}")
        return None

def query_ollama_vision(image_paths, prompt, model=DEFAULT_MODEL):
    """
    Send a request to Ollama Vision model with one or more images.
    
    Args:
        image_paths: List of paths to image files
        prompt: Text prompt to send with the images
        model: Ollama model name (default: llama3.2-vision)
        
    Returns:
        Response text from the model
    """
    if not isinstance(image_paths, list):
        image_paths = [image_paths]
    
    # Encode all images to base64
    base64_images = []
    for img_path in image_paths:
        encoded = encode_image_to_base64(img_path)
        if encoded:
            base64_images.append(encoded)
    
    if not base64_images:
        return "Error: Failed to encode any images"
    
    # Create the API request payload
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": base64_images
            }
        ],
        "stream": False
    }
    
    try:
        # Make the API request to Ollama
        logger.info(f"Sending request to Ollama API with {len(base64_images)} images...")
        response = requests.post(
            f"{OLLAMA_API_BASE}/chat",
            json=payload
        )
        
        # Check response status
        if response.status_code == 200:
            return response.json()
        else:
            return f"Error: API request failed with status {response.status_code} - {response.text}"
            
    except Exception as e:
        return f"Error: {str(e)}"

def print_model_info(model=DEFAULT_MODEL):
    """Print information about the Ollama model"""
    try:
        response = requests.get(f"{OLLAMA_API_BASE}/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            for model_info in models:
                if model_info.get("name") == model:
                    print(f"Model: {model_info.get('name')}")
                    print(f"Size: {model_info.get('size')}")
                    print(f"Modified: {model_info.get('modified')}")
                    print(f"Full name: {model_info.get('details', {}).get('format')}")
                    return True
                    
            print(f"Model '{model}' not found on your Ollama server.")
            print("Available models:")
            for model_info in models:
                print(f"- {model_info.get('name')}")
            return False
        else:
            print(f"Error: Could not get model info, status {response.status_code}")
            return False
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

def check_ollama_server():
    """Check if Ollama server is running"""
    try:
        response = requests.get(f"{OLLAMA_API_BASE}/tags")
        return response.status_code == 200
    except:
        return False

def main():
    """Main function to run the Ollama vision test"""
    # Check if Ollama server is running
    if not check_ollama_server():
        logger.error("Ollama server is not running. Please start it with 'ollama serve' command.")
        return
    
    parser = argparse.ArgumentParser(description="Test Ollama Vision models with local images")
    parser.add_argument('--image', '-i', type=str, nargs='+', required=True, 
                        help='Path(s) to image file(s) to analyze')
    parser.add_argument('--prompt', '-p', type=str, default="What is in this image? Describe it in detail.", 
                        help='Text prompt to send with the image(s)')
    parser.add_argument('--model', '-m', type=str, default=DEFAULT_MODEL, 
                        help=f'Ollama vision model to use (default: {DEFAULT_MODEL})')
    parser.add_argument('--output', '-o', type=str, 
                        help='Optional path to save the response to a JSON file')
    
    args = parser.parse_args()
    
    # Print info about the selected model
    print("\n=== Ollama Vision Model Test ===")
    if not print_model_info(args.model):
        logger.warning(f"Proceeding with model {args.model} anyway.")
    
    # Send images to the model
    images = []
    for img_path in args.image:
        if os.path.exists(img_path):
            images.append(img_path)
            logger.info(f"Added image: {img_path}")
        else:
            logger.warning(f"Image not found: {img_path}")
    
    if not images:
        logger.error("No valid images provided. Exiting.")
        return
    
    # Get image descriptions from Ollama
    logger.info(f"Sending {len(images)} images to {args.model} with prompt: '{args.prompt}'")
    result = query_ollama_vision(images, args.prompt, args.model)
    
    # Print the response
    print("\n=== Model Response ===")
    if isinstance(result, str):
        print(result)  # Error message
    else:
        # Extract the response content
        message = result.get("message", {})
        content = message.get("content", "No content in response")
        print(content)
        
        # Save to file if requested
        if args.output:
            try:
                with open(args.output, 'w') as f:
                    json.dump(result, f, indent=2)
                logger.info(f"Response saved to {args.output}")
            except Exception as e:
                logger.error(f"Failed to save response: {e}")

if __name__ == "__main__":
    main()