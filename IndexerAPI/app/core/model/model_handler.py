import base64
import logging
from typing import List, Dict, Any, Union
import os
from openai import AsyncOpenAI
import httpx
import asyncio
import json
from app.config import settings

logger = logging.getLogger(__name__)


class ModelHandler:
    def __init__(self, api_key: str = None, api_base: str = None):
        self.embedding_api_key = api_key or settings.EMBEDDING_API_KEY
        self.embedding_api_base = api_base or settings.EMBEDDING_API_BASE
        self.inference_api_key = api_key or settings.INFERENCE_API_KEY
        self.inference_api_base = api_base or settings.INFERENCE_API_BASE
        http_client = httpx.AsyncClient(timeout=7200.0)
        self.embedding_client = AsyncOpenAI(api_key=self.embedding_api_key, base_url=self.embedding_api_base, http_client=http_client)
        self.inference_client = AsyncOpenAI(api_key=self.inference_api_key, base_url=self.inference_api_base, http_client=http_client)
    
    async def generate_alt_text(self, image_base64: str, model: str = "gemma3:12b-it-q8_0") -> str:
        if not image_base64:
            logger.warning("Empty image_base64 provided for alt text generation")
            return ""
        
        system_prompt = """
        You are an AI assistant whose job is to generate rich, descriptive alt text for the provided documents in a multimodal RAG pipeline. For each input document-whether it comes from a PDF page, a webpage screenshot, a DOCX export, or a standalone photo—you will produce concise, context-aware alt text that:

        1. Identifies and names all salient entities, objects, and text visible in the document.
        2. Describes relationships, actions, or interactions depicted.
        3. Conveys any relevant context or setting needed for understanding the document.
        4. Remains clear and unambiguous, suitable for embedding alongside this document to provide downstream models with full context.

        Note: The provided image can be a screenshot of a webpage, a PDF page, or any other image format. Your task is to generate alt text that accurately describes the content of the document.

        Your alt text will be attached to each document before indexing, ensuring that the multimodal retrieval system can leverage both visual and textual cues effectively.
        """

        try:
            response = await self.inference_client.chat.completions.create(
                model=model,
                messages=[  
                            {
                                "role": "system",
                                "content": system_prompt,
                            },   
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": "Generate alt text for the following document. Just the alt text, no other text like 'Here is the alt text:',etc"},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": "data:image/png;base64," + image_base64,
                                        }
                                    },
                                ],
                            }
                        ],
            )
            alt_text = response.choices[0].message.content.strip()
            return alt_text
            
        except Exception as e:
            logger.error(f"Error generating alt text: {str(e)}")
            raise

    async def embed_text(self, texts: List[str], model: str = "nomic-ai/nomic-embed-multimodal-3b") -> List[List[float]]:
        if not texts:
            logger.warning("Empty texts list provided for embedding")
            return []

        try:
            response = await self.embedding_client.embeddings.create(
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

    async def embed_image(self, image_texts: List[Dict[str, str]], model: str = "nomic-ai/nomic-embed-multimodal-3b") -> List[List[float]]:
        if not image_texts:
            logger.warning("Empty image_texts list provided for embedding")
            return []
        
        logger.info(f"Processing {len(image_texts)} images for embedding")

        try:
            response = await self.embedding_client.embeddings.create(
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