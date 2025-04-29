import base64
import logging
from typing import List, Dict, Any, Union
import os
from openai import OpenAI
import json

logger = logging.getLogger(__name__)


class ModelHandler:
    def __init__(self, api_key: str = None, api_base: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.api_base = api_base or os.getenv("OPENAI_API_BASE")
        self.client = OpenAI(api_key=self.api_key, base_url=self.api_base)
        logger.info(f"ModelHandler initialized with API base: {self.api_base}")

    def embed_text(self, texts: List[str], model: str = "nomic-ai/colnomic-embed-multimodal-3b", batch_size: int = 4) -> List[List[float]]:
        if not texts:
            logger.warning("Empty texts list provided for embedding")
            return []

        all_embeddings = []
        num_items = len(texts)
        logger.info(f"Generating embeddings for {num_items} text items using model: {model} with batch size: {batch_size}")

        try:
            for i in range(0, num_items, batch_size):
                batch = texts[i:min(i + batch_size, num_items)]
                logger.info(f"Processing batch {i // batch_size + 1}/{(num_items + batch_size - 1) // batch_size} with {len(batch)} items")
                
                response = self.client.embeddings.create(
                    input=batch,
                    model=model,
                    encoding_format="float"
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                logger.info(f"Successfully generated {len(batch_embeddings)} embeddings for this batch")

            logger.info(f"Successfully generated a total of {len(all_embeddings)} text embeddings")
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Error generating text embeddings: {str(e)}")
            raise

    def embed_image(self, image_texts: List[Dict[str, str]], model: str = "nomic-ai/colnomic-embed-multimodal-3b", batch_size: int = 4) -> List[List[float]]:
        if not image_texts:
            logger.warning("Empty image_texts list provided for embedding")
            return []

        all_embeddings = []
        num_items = len(image_texts)
        logger.info(f"Generating embeddings for {num_items} image-text pairs using model: {model} with batch size: {batch_size}")

        try:
            for i in range(0, num_items, batch_size):
                batch = image_texts[i:min(i + batch_size, num_items)]
                logger.info(f"Processing batch {i // batch_size + 1}/{(num_items + batch_size - 1) // batch_size} with {len(batch)} items")
                
                response = self.client.embeddings.create(
                    input=batch,
                    model=model,
                    encoding_format="float"
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                logger.info(f"Successfully generated {len(batch_embeddings)} embeddings for this batch")

            logger.info(f"Successfully generated a total of {len(all_embeddings)} image embeddings")
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Error generating image embeddings: {str(e)}")
            raise

    def encode_image_to_base64(self, image_path: str) -> str:
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encoding image to base64: {str(e)}")
            raise


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    test_payload = [
        {
            "image": "iVBORw0KGgoAAAANSUhEUgAAAIAAAACACAIAAABMXPacAAABLUlEQVR4nO3RQREAIAzAMMC/501GHjQKetc7MyfO0wG/awDWAKwBWAOwBmANwBqANQBrANYArAFYA7AGYA3AGoA1AGsA1gCsAVgDsAZgDcAagDUAawDWAKwBWAOwBmANwBqANQBrANYArAFYA7AGYA3AGoA1AGsA1gCsAVgDsAZgDcAagDUAawDWAKwBWAOwBmANwBqANQBrANYArAFYA7AGYA3AGoA1AGsA1gCsAVgDsAZgDcAagDUAawDWAKwBWAOwBmANwBqANQBrANYArAFYA7AGYA3AGoA1AGsA1gCsAVgDsAZgDcAagDUAawDWAKwBWAOwBmANwBqANQBrANYArAFYA7AGYA3AGoA1AGsA1gCsAVgDsAZgDcAagDUAawDWAKwBWAOwBmANwBqANQBrANYArAFYA7AGYA3AGoA1AGsA1gCsAVgDsAZgDcAagDUAawBbgZgP9Ag5IZgAAAABJRU5ErkJggg==",
            "text": "This is a sample image"
        }
    ]
    
    try:
        model_handler = ModelHandler()
        
        print("Testing image embedding...")
        embeddings = model_handler.embed_image(test_payload)
        
        print(f"Generated {len(embeddings)} embeddings")
        if embeddings:
            print(f"Embedding dimensions: {len(embeddings[0])}")
            print(f"First few values of embedding: {embeddings[0][:5]}")
        
        print("\nTesting text embedding...")
        text_embeddings = model_handler.embed_text(["This is a sample text for embedding."])
        
        if text_embeddings:
            print(f"Text embedding dimensions: {len(text_embeddings[0])}")
            print(f"First few values of text embedding: {text_embeddings[0][:5]}")
            
    except Exception as e:
        logger.error(f"Error during testing: {str(e)}")
        print(f"Test failed: {str(e)}")