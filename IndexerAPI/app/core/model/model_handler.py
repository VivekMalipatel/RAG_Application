import base64
import logging
from typing import List, Dict, Any, Union
import os
from openai import AsyncOpenAI
import httpx
import asyncio
import json

logger = logging.getLogger(__name__)


class ModelHandler:
    def __init__(self, api_key: str = None, api_base: str = None):
        self.api_key = api_key or os.getenv("EMBEDDING_API_KEY")
        self.api_base = api_base or os.getenv("EMBEDDING_API_BASE")
        http_client = httpx.AsyncClient(timeout=7200.0)
        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.api_base, http_client=http_client)
        logger.info(f"ModelHandler initialized with API base: {self.api_base} and async client")

    async def embed_text(self, texts: List[str], model: str = "nomic-ai/colnomic-embed-multimodal-3b") -> List[List[float]]:
        if not texts:
            logger.warning("Empty texts list provided for embedding")
            return []

        logger.info(f"Generating embeddings for {len(texts)} text items using model: {model}")

        try:
            response = await self.client.embeddings.create(
                input=texts,
                model=model,
                encoding_format="float"
            )
            
            embeddings = [item.embedding for item in response.data]
            logger.info(f"Successfully generated {len(embeddings)} text embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating text embeddings: {str(e)}")
            raise

    async def embed_image(self, image_texts: List[Dict[str, str]], model: str = "nomic-ai/colnomic-embed-multimodal-3b") -> List[List[float]]:
        if not image_texts:
            logger.warning("Empty image_texts list provided for embedding")
            return []

        logger.info(f"Generating embeddings for {len(image_texts)} image-text pairs using model: {model}")

        try:
            response = await self.client.embeddings.create(
                input=image_texts,
                model=model,
                encoding_format="float"
            )
            
            embeddings = [item.embedding for item in response.data]
            logger.info(f"Successfully generated {len(embeddings)} image embeddings")
            return embeddings
            
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


async def main_test():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    test_payload = [
        {
            "image": "iVBORw0KGgoAAAANSUhEUgAAAIAAAACACAIAAABMXPacAAABLUlEQVR4nO3RQREAIAzAMMC/501GHjQKetc7MyfO0wG/awDWAKwBWAOwBmANwBqANQBrANYArAFYA7AGYA3AGoA1AGsA1gCsAVgDsAZgDcAagDUAawDWAKwBWAOwBmANwBqANQBrANYArAFYA7AGYA3AGoA1AGsA1gCsAVgDsAZgDcAagDUAawDWAKwBWAOwBmANwBqANQBrANYArAFYA7AGYA3AGoA1AGsA1gCsAVgDsAZgDcAagDUAawBbgZgP9Ag5IZgAAAABJRU5ErkJggg==",
            "text": "This is a sample image"
        }
    ]
    
    try:
        model_handler = ModelHandler()
        
        print("Testing image embedding...")
        embeddings = await model_handler.embed_image(test_payload)
        
        print(f"Generated {len(embeddings)} embeddings")
        if embeddings:
            print(f"Embedding dimensions: {len(embeddings[0])}")
            print(f"First few values of embedding: {embeddings[0][:5]}")
        
        print("\nTesting text embedding...")
        text_embeddings = await model_handler.embed_text(["This is a sample text for embedding."])
        
        if text_embeddings:
            print(f"Text embedding dimensions: {len(text_embeddings[0])}")
            print(f"First few values of text embedding: {text_embeddings[0][:5]}")
            
    except Exception as e:
        logger.error(f"Error during testing: {str(e)}")
        print(f"Test failed: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main_test())