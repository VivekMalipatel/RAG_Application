"""
Test script for the Model Router API handler integration.
"""

import os
import sys
import asyncio
from pathlib import Path

# Add the parent directory to the system path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.model import ModelClient, TextGenerator, EmbeddingGenerator

async def test_model_client():
    """Test basic model client functionality"""
    print("\n=== Testing Model Client ===")
    
    # Initialize the model client
    client = ModelClient()
    print(f"Model client initialized with base URL: {client.config.base_url}")
    
    # Check API health
    health = client.health_check()
    print(f"Model Router API health check: {'OK' if health else 'Failed'}")
    
    if not health:
        print("Please ensure Model Router API is running")
        return False
        
    # List available models
    models = client.get_available_models()
    print(f"Available models: {len(models)}")
    for model in models[:5]:  # Show first 5 models only
        print(f"  - {model.get('id', 'unknown')}")
    
    return True

async def test_text_generation():
    """Test text generation capabilities"""
    print("\n=== Testing Text Generation ===")
    
    # Initialize text generator
    generator = TextGenerator()
    print(f"Text generator initialized with model: {generator.model}")
    
    # Generate text
    prompt = "Explain the concept of Retrieval-Augmented Generation (RAG) in 3 sentences."
    system_message = "You are a helpful assistant providing concise explanations."
    
    print(f"Generating text for prompt: '{prompt}'")
    result = await generator.generate_text(prompt, system_message)
    
    if result["success"]:
        print("\nGenerated Text:")
        print(result["content"])
        print("\nUsage statistics:")
        print(f"  - Prompt tokens: {result['usage']['prompt_tokens']}")
        print(f"  - Completion tokens: {result['usage']['completion_tokens']}")
        print(f"  - Total tokens: {result['usage']['total_tokens']}")
        return True
    else:
        print(f"Error generating text: {result.get('error', 'Unknown error')}")
        return False

async def test_structured_output():
    """Test structured output generation"""
    print("\n=== Testing Structured Output Generation ===")
    
    # Initialize text generator
    generator = TextGenerator()
    
    # Define a schema for document analysis
    schema = {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "The title or main topic of the text"
            },
            "summary": {
                "type": "string",
                "description": "A brief summary of the content (1-2 sentences)"
            },
            "key_points": {
                "type": "array",
                "description": "Key points extracted from the content",
                "items": {
                    "type": "string"
                }
            },
            "sentiment": {
                "type": "string",
                "enum": ["positive", "neutral", "negative"],
                "description": "The overall sentiment of the document"
            }
        },
        "required": ["title", "summary", "key_points", "sentiment"]
    }
    
    # Sample text to analyze
    text = """
    # Retrieval-Augmented Generation
    
    Retrieval-Augmented Generation (RAG) is an AI framework that combines the strengths of 
    retrieval-based and generation-based approaches. It enhances large language models by 
    providing them with access to external knowledge sources. This allows models to generate 
    more accurate, factual responses while reducing hallucinations. RAG is particularly 
    effective for domain-specific applications where accurate information retrieval is crucial.
    """
    
    prompt = f"Analyze the following text and extract key information: {text}"
    
    print(f"Generating structured output for text analysis")
    result = await generator.generate_structured_output(prompt, schema)
    
    if result["success"]:
        print("\nStructured Output:")
        import json
        print(json.dumps(result["data"], indent=2))
        return True
    else:
        print(f"Error generating structured output: {result.get('error', 'Unknown error')}")
        return False

async def test_embeddings():
    """Test embedding generation capabilities"""
    print("\n=== Testing Embedding Generation ===")
    
    # Initialize embedding generator
    embedder = EmbeddingGenerator()
    print(f"Embedding generator initialized with model: {embedder.model}")
    
    # Generate an embedding for a single text
    text = "Retrieval-Augmented Generation is a technique that enhances LLMs with external knowledge."
    print(f"Generating embedding for text: '{text[:50]}...'")
    
    result = await embedder.generate_embedding(text)
    
    if result["success"]:
        embedding = result["embedding"]
        print(f"Generated embedding with {result['dimensions']} dimensions")
        print(f"First 5 values: {embedding[:5]}")
        print(f"Token usage: {result['usage']['total_tokens']}")
        
        # Test batch processing
        texts = [
            "Retrieval-Augmented Generation combines retrieval and generation techniques.",
            "Vector databases store and efficiently search through vector embeddings.",
            "Large Language Models can generate human-like text but may hallucinate facts."
        ]
        
        print(f"\nGenerating embeddings for {len(texts)} texts")
        batch_result = await embedder.generate_embeddings_batch(texts)
        
        if batch_result["success"]:
            print(f"Generated {batch_result['count']} embeddings with {batch_result['dimensions']} dimensions each")
            print(f"Total token usage: {batch_result['usage']['total_tokens']}")
            
            # Test cosine similarity
            similarity = embedder.cosine_similarity(
                batch_result["embeddings"][0], 
                batch_result["embeddings"][1]
            )
            print(f"\nCosine similarity between first two embeddings: {similarity:.4f}")
            return True
        else:
            print(f"Error generating batch embeddings: {batch_result.get('error', 'Unknown error')}")
            return False
    else:
        print(f"Error generating embedding: {result.get('error', 'Unknown error')}")
        return False

async def run_all_tests():
    """Run all tests sequentially"""
    print("=== Running Model Router API Handler Tests ===")
    
    # Test the model client first
    client_ok = await test_model_client()
    if not client_ok:
        print("Model client test failed, skipping remaining tests")
        return
    
    # Test text generation
    await test_text_generation()
    
    # Test structured output
    await test_structured_output()
    
    # Test embeddings
    await test_embeddings()
    
    print("\n=== All Tests Complete ===")

if __name__ == "__main__":
    asyncio.run(run_all_tests())