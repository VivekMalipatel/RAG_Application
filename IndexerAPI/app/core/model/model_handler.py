import base64
import logging
from typing import List, Dict, Any, Optional
import json
from openai import AsyncOpenAI
import aiohttp
import asyncio
from pydantic import BaseModel
from openai.resources.chat.completions.completions import ChatCompletion, ParsedChatCompletion
from config import settings

logger = logging.getLogger(__name__)

_global_model_handler: Optional['ModelHandler'] = None

class EmbeddingRateLimiter:
    def __init__(self, max_concurrent_requests: int = 1):
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.request_queue = asyncio.Queue()
        self.processing_tasks = set()
        self._shutdown = False
    
    async def process_queue(self, embed_func):
        while not self._shutdown:
            try:
                request_data = await asyncio.wait_for(self.request_queue.get(), timeout=1.0)
                if request_data is None:
                    break
                
                messages, future = request_data
                task = asyncio.create_task(self._process_request(embed_func, messages, future))
                self.processing_tasks.add(task)
                task.add_done_callback(self.processing_tasks.discard)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in queue processor: {e}")
    
    async def _process_request(self, embed_func, messages, future):
        async with self.semaphore:
            try:
                result = await embed_func(messages)
                if not future.cancelled():
                    future.set_result(result)
            except Exception as e:
                if not future.cancelled():
                    future.set_exception(e)
    
    async def add_request(self, messages: List[dict]) -> List[List[float]]:
        future = asyncio.Future()
        await self.request_queue.put((messages, future))
        return await future
    
    async def shutdown(self):
        self._shutdown = True
        await self.request_queue.put(None)
        if self.processing_tasks:
            await asyncio.gather(*self.processing_tasks, return_exceptions=True)

class ChatRateLimiter:
    def __init__(self, max_concurrent_requests: int = 1):
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.request_queue = asyncio.Queue()
        self.processing_tasks = set()
        self._shutdown = False
    
    async def process_queue(self, regular_func, structured_func):
        while not self._shutdown:
            try:
                request_data = await asyncio.wait_for(self.request_queue.get(), timeout=1.0)
                if request_data is None:
                    break
                
                request_params, is_structured, future = request_data
                chat_func = structured_func if is_structured else regular_func
                task = asyncio.create_task(self._process_request(chat_func, request_params, future))
                self.processing_tasks.add(task)
                task.add_done_callback(self.processing_tasks.discard)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in chat queue processor: {e}")
    
    async def _process_request(self, chat_func, request_params, future):
        async with self.semaphore:
            try:
                result = await chat_func(**request_params)
                if not future.cancelled():
                    future.set_result(result)
            except Exception as e:
                if not future.cancelled():
                    future.set_exception(e)
    
    async def add_request(self, is_structured=False, **request_params):
        future = asyncio.Future()
        await self.request_queue.put((request_params, is_structured, future))
        return await future
    
    async def shutdown(self):
        self._shutdown = True
        await self.request_queue.put(None)
        if self.processing_tasks:
            await asyncio.gather(*self.processing_tasks, return_exceptions=True)

class EntitySchema(BaseModel):
    id: str
    text: str 
    entity_type: str
    entity_profile: str

class RelationSchema(BaseModel):
    source: str 
    target: str 
    relation_type: str
    relation_profile: str 

class EntityRelationSchema(BaseModel):
    entities: List[EntitySchema]
    relationships: List[RelationSchema]

class ColumnSchema(BaseModel):
    column_name: str
    column_profile: str

class ColumnProfilesSchema(BaseModel):
    columns: List[ColumnSchema]

class ModelHandler:
    def __init__(self, api_key: str = None, api_base: str = None):
        self.embedding_api_key = api_key or settings.EMBEDDING_API_KEY
        self.embedding_api_base = api_base or settings.EMBEDDING_API_BASE
        self.inference_api_key = api_key or settings.INFERENCE_API_KEY
        self.inference_api_base = api_base or settings.INFERENCE_API_BASE
        self.embedding_client = AsyncOpenAI(api_key=self.embedding_api_key, base_url=self.embedding_api_base, timeout=settings.EMBEDDING_CLIENT_TIMEOUT, max_retries=settings.RETRIES)
        self.inference_client = AsyncOpenAI(api_key=self.inference_api_key, base_url=self.inference_api_base, timeout=settings.EMBEDDING_CLIENT_TIMEOUT, max_retries=settings.RETRIES)
        
        self.embedding_rate_limiter = EmbeddingRateLimiter(max_concurrent_requests=settings.EMBEDDING_CONCURRENT_REQUESTS)
        self.chat_rate_limiter = ChatRateLimiter(max_concurrent_requests=settings.INFERENCE_CONCURRENT_REQUESTS)
        
        self.embedding_queue_processor_task = None
        self.chat_queue_processor_task = None
        self.text_description_queue_processor_task = None
        self.entity_extraction_queue_processor_task = None
        
        self._start_queue_processors()
    
    def _start_queue_processors(self):
        if self.embedding_queue_processor_task is None or self.embedding_queue_processor_task.done():
            self.embedding_queue_processor_task = asyncio.create_task(
                self.embedding_rate_limiter.process_queue(self._embed_internal)
            )
        
        if self.chat_queue_processor_task is None or self.chat_queue_processor_task.done():
            self.chat_queue_processor_task = asyncio.create_task(
                self.chat_rate_limiter.process_queue(self._chat_completion_internal, self._structured_chat_completion_internal)
            )
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.shutdown()
    
    async def shutdown(self):
        if self.embedding_rate_limiter:
            await self.embedding_rate_limiter.shutdown()
        
        if self.chat_rate_limiter:
            await self.chat_rate_limiter.shutdown()
            
        tasks_to_cancel = [
            self.embedding_queue_processor_task,
            self.chat_queue_processor_task,
        ]
        
        for task in tasks_to_cancel:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
    
    async def generate_text_description(self, image_base64: str) -> str:
        if not image_base64:
            logger.warning("Empty image_base64 provided for text generation")
            return ""
        
        system_prompt = """You are an AI assistant whose job is to generate rich, descriptive text for the provided documents in a multimodal RAG pipeline. For each input document-whether it comes from a PDF page, a webpage screenshot, a DOCX export, or a standalone photo—you will produce concise, context-aware text that:

        1. Identifies and names all salient Named and Unnamed entities, objects, and data visible in the document.
        2. Describes relationships, actions, or interactions depicted.
        3. Conveys any relevant context or setting needed for understanding the document.
        4. Remains clear and unambiguous, suitable for embedding alongside this document to provide downstream models with full context.

        Note: The provided image can be a screenshot of a webpage, a PDF page, or any other image format. Your task is to generate text that accurately describes the content of the document.

        Your text will be attached to each document before indexing, ensuring that the multimodal retrieval system can leverage both visual and textual cues effectively.
        """
        
        try:
            response : ChatCompletion = await self.chat_completion(
                model=settings.INFERENCE_MODEL,
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
                        ]
            )
            if (response and 
                hasattr(response, 'choices') and 
                response.choices and 
                len(response.choices) > 0 and
                hasattr(response.choices[0], 'message') and
                hasattr(response.choices[0].message, 'content') and
                response.choices[0].message.content):
                alt_text = response.choices[0].message.content.strip()
                return alt_text
            else:
                logger.error("No valid response received for text description")
                return ""
        except Exception as e:
            logger.error(f"Error generating alt text: {str(e)}")
            return ""

    async def embed(self, messages: List[dict]) -> List[List[float]]:
        self._start_queue_processors()
        return await self.embedding_rate_limiter.add_request(messages)

    async def _embed_internal(self, messages: List[dict]) -> List[List[float]]:

        url = f"{self.embedding_api_base}/embeddings"
        headers = {
            "Authorization": "Bearer " + self.embedding_api_key,
            "Content-Type": "application/json"
        }
        
        settings.RETRIES = 1 if settings.RETRIES < 1 else settings.RETRIES
        for attempt in range(settings.RETRIES):
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=settings.EMBEDDING_CLIENT_TIMEOUT)) as session:
                    async with session.post(
                        url,
                        headers=headers,
                        json={
                            "model": settings.EMBEDDING_MODEL,
                            "messages": messages,
                            "encoding_format": "float"
                        }
                    ) as response:
                        response.raise_for_status()
                        data = await response.json()
                        embeddings = [item["embedding"] for item in data["data"]]
                        return embeddings
            except Exception as e:
                if attempt == settings.RETRIES - 1:
                    logger.error(f"Error generating image embeddings after {settings.RETRIES} attempts: {str(e)}")
                    zero_embedding = [0.0] * settings.EMBEDDING_DIMENSIONS
                    return [zero_embedding for _ in messages]
                else:
                    logger.warning(f"Attempt {attempt + 1} failed for image embeddings: {str(e)}. Retrying...")
                    await asyncio.sleep(settings.RETRY_DELAY)
    
    async def _chat_completion_internal(self, **kwargs):
        try:
            response = await self.inference_client.chat.completions.create(**kwargs)
            return response
        except Exception as e:
            logger.error(f"Error in chat completion: {str(e)}")
            return None
    
    async def _structured_chat_completion_internal(self, **kwargs):
        try:
            response = await self.inference_client.beta.chat.completions.parse(**kwargs)
            return response
        except Exception as e:
            logger.error(f"Error in structured chat completion: {str(e)}")
            return None
    
    async def chat_completion(self, **kwargs):
        self._start_queue_processors()
        return await self.chat_rate_limiter.add_request(is_structured=False, **kwargs)
    
    async def structured_chat_completion(self, **kwargs):
        self._start_queue_processors()
        return await self.chat_rate_limiter.add_request(is_structured=True, **kwargs)
    
    async def embed_text(self, texts: List[str]) -> List[List[float]]:
        try:
            response = await self.embedding_client.embeddings.create(
                input=texts,
                model=settings.EMBEDDING_MODEL,
                encoding_format="float"
            )
            
            embeddings = [item.embedding for item in response.data]
            logger.info(f"Successfully generated {len(embeddings)} text embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating text embeddings: {str(e)}")
            zero_embedding = [0.0] * settings.EMBEDDING_DIMENSIONS
            return [zero_embedding for _ in texts]

    async def extract_entities_relationships(self, messages: List[dict]) -> Dict[str, Any]:
        try:
            extraction_prompt = """Extract entities and relationships (max 15 each) from the content for document analysis.

            REQUIREMENTS:
            1. Extract key information with high precision
            2. Use lowercase_underscore IDs: "John Smith" -> "john_smith"
            3. Provide detailed profiles and contextual relationships
            4. Handle coreference resolution (pronouns -> actual names)
            5. Consider visual elements in images (charts, diagrams, etc.)
            6. Be concise - max 15 top entities and relationships (less than 10000 characters total)

            OUTPUT FORMAT - Return ONLY valid JSON in this exact format:
            {
                "entities": [
                    {
                        "id": "entity_id_lowercase_with_underscores",
                        "text": "Original entity text",
                        "entity_type": "ENTITY_TYPE",
                        "entity_profile": "Detailed description of role, context, and significance"
                    }
                ],
                "relationships": [
                    {
                        "source": "source_entity_id",
                        "target": "target_entity_id",
                        "relation_type": "RELATIONSHIP_TYPE",
                        "relation_profile": "Description of how and why entities are related"
                    }
                ]
            }

            SOME OF THE EXAMPLES OF ENTITY TYPES:
            PERSON, ORGANIZATION, LOCATION, DOCUMENT, IDENTIFIER, CONCEPT, FINANCIAL, DATE_TIME, REQUIREMENT, POSITION_TITLE, CONTACT_INFO, ASSET, PROCESS, CLASSIFICATION, PRODUCT_SERVICE, METRIC

            SOME OF THE EXAMPLES OF RELATIONSHIP TYPES:
            WORKS_FOR, MANAGES, REPORTS_TO, COLLABORATES_WITH, ASSOCIATED_WITH, LOCATED_AT, VALID_FROM/UNTIL, RESPONSIBLE_FOR, AUTHORED_BY, REFERENCES, CONTAINS, PARTICIPATES_IN, RELATED_TO, DEPENDS_ON, ASSIGNED_TO, DESCRIBES, BELONGS_TO, COMMUNICATES_WITH

            Focus on extracting the most significant entities and relationships that capture the document's key information, structure, and purpose.
            IMPORTANT: Return ONLY the JSON object, no additional text or explanation.
            """

            system_message = {"role": "system", "content": extraction_prompt}

            has_image = len(messages[0]["content"]) > 1
            
            if has_image:
                image_messages = [
                    {
                        "role": "user", 
                        "content": [messages[0]["content"][0]]
                    }
                ]
                text_messages = [
                    {
                        "role": "user", 
                        "content": [messages[0]["content"][1]]
                    }
                ]
            else:
                image_messages = []
                text_messages = [
                    {
                        "role": "user", 
                        "content": [messages[0]["content"][0]]
                    }
                ]

            try:
                if has_image:
                    response : ChatCompletion = await self.chat_completion(
                        model=settings.INFERENCE_MODEL,
                        messages=[system_message] + image_messages,
                        max_completion_tokens=settings.INFERNECE_STRUCTURED_OUTPUTS_MAX_TOKENS,
                    )
                else:
                    raise ValueError("No image content available, falling back to text-based extraction")
                
                if (response and 
                    hasattr(response, 'choices') and 
                    response.choices and 
                    len(response.choices) > 0 and
                    hasattr(response.choices[0], 'message') and
                    hasattr(response.choices[0].message, 'content') and
                    response.choices[0].message.content):
                    
                    content = response.choices[0].message.content.strip()
                    
                    if content.startswith('```json'):
                        content = content[7:]
                    if content.endswith('```'):
                        content = content[:-3]
                    content = content.strip()
                    
                    try:
                        parsed_result = json.loads(content)
                        entities = parsed_result.get("entities", [])
                        relationships = parsed_result.get("relationships", [])
                        
                        for entity in entities:
                            entity["id"] = entity["id"].lower().replace(" ", "_").replace("-", "_")
                        
                        for rel in relationships:
                            rel["source"] = rel["source"].lower().replace(" ", "_").replace("-", "_")
                            rel["target"] = rel["target"].lower().replace(" ", "_").replace("-", "_")
                        
                        return {"entities": entities, "relationships": relationships}
                    except json.JSONDecodeError as json_e:
                        logger.warning(f"Failed to parse JSON from image response: {json_e}. Content: {content[:500]}")
                        raise ValueError("Failed to parse image-based JSON response")
                else:
                    raise ValueError("Failed to get valid response from image-based extraction")
                    
            except Exception as e:
                logger.warning(f"Image-based entity extraction failed: {e}. Falling back to text-based extraction.")
                
                try:
                    response : ChatCompletion = await self.chat_completion(
                        model=settings.REASONING_MODEL,
                        messages=[system_message] + text_messages,
                        max_completion_tokens=settings.REASONING_STRUCTURED_OUTPUTS_MAX_TOKENS,
                    )
                
                    if (response and 
                        hasattr(response, 'choices') and 
                        response.choices and 
                        len(response.choices) > 0 and
                        hasattr(response.choices[0], 'message') and
                        hasattr(response.choices[0].message, 'content') and
                        response.choices[0].message.content):
                        
                        content = response.choices[0].message.content.strip()
                        
                        if content.startswith('```json'):
                            content = content[7:]
                        if content.endswith('```'):
                            content = content[:-3]
                        content = content.strip()
                        
                        try:
                            parsed_result = json.loads(content)
                            entities = parsed_result.get("entities", [])
                            relationships = parsed_result.get("relationships", [])
                            
                            for entity in entities:
                                entity["id"] = entity["id"].lower().replace(" ", "_").replace("-", "_")
                            
                            for rel in relationships:
                                rel["source"] = rel["source"].lower().replace(" ", "_").replace("-", "_")
                                rel["target"] = rel["target"].lower().replace(" ", "_").replace("-", "_")
                            
                            return {"entities": entities, "relationships": relationships}
                        except json.JSONDecodeError as json_e:
                            logger.error(f"Failed to parse JSON from text response: {json_e}. Content: {content[:500]}")
                            return {"entities": [], "relationships": []}
                    else:
                        logger.error("Both image and text-based entity extraction failed")
                        return {"entities": [], "relationships": []}
                        
                except Exception as text_e:
                    logger.error(f"Text-based entity extraction also failed: {text_e}")
                    return {"entities": [], "relationships": []}
            
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return {"entities": [], "relationships": []}

    async def embed_entity_relationship_profiles(self, entities: List[Dict[str, Any]], relationships: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        try:
            entity_profiles = [entity.get("entity_profile", "") for entity in entities]
            relationship_profiles = [rel.get("relation_profile", "") for rel in relationships]
            
            all_profiles = entity_profiles + relationship_profiles
            
            if not all_profiles:
                return entities, relationships
            
            embeddings = await self.embed_text(all_profiles)
            
            entity_embeddings = embeddings[:len(entity_profiles)]
            relationship_embeddings = embeddings[len(entity_profiles):]
            
            for i, entity in enumerate(entities):
                if i < len(entity_embeddings):
                    entity["embedding"] = entity_embeddings[i]
            
            for i, rel in enumerate(relationships):
                if i < len(relationship_embeddings):
                    rel["embedding"] = relationship_embeddings[i]
            
            return entities, relationships
            
        except Exception as e:
            logger.error(f"Error embedding entity/relationship profiles: {e}")
            return entities, relationships

    async def generate_structured_summary(self, dataframe_text: str) -> str:
        try:

            system_prompt = """You are an AI assistant that analyzes structured data and provides comprehensive summaries. 
            Given a tabular dataset representation, provide a clear, informative summary that includes:
            1. Overall purpose and content of the dataset
            2. Key patterns and insights
            3. Data quality observations
            4. Notable trends or anomalies
            5. Potential use cases or applications
            
            Keep the summary concise but comprehensive, suitable for embedding and retrieval."""

            user_prompt = f"Analyze this structured dataset and provide a comprehensive summary:\n\n{dataframe_text}"

            response :ChatCompletion = await self.chat_completion(
                model=settings.INFERENCE_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            if (response and 
                hasattr(response, 'choices') and 
                response.choices and 
                len(response.choices) > 0 and
                hasattr(response.choices[0], 'message') and
                hasattr(response.choices[0].message, 'content') and
                response.choices[0].message.content):
                return response.choices[0].message.content.strip()
            else:
                logger.error("No valid response received for structured summary")
                return "Unable to generate summary due to API error"
            
        except Exception as e:
            logger.error(f"Error generating structured summary: {e}")
            return f"Error generating summary: {str(e)}"

    async def generate_column_profiles(self, dataframe_text: str) -> List[Dict[str, str]]:
        try:

            system_prompt = """You are an AI assistant that analyzes structured data columns. 
            Given a tabular dataset, analyze each column and provide detailed profiles.
            
            For each column, provide:
            1. Column name (exactly as it appears in the dataset)
            2. Comprehensive column profile including:
               - Data type and format
               - Content description and meaning
               - Data quality assessment
               - Statistical characteristics (if applicable)
               - Potential relationships with other columns
               - Business or analytical significance
               - Common patterns or unique characteristics
            
            Extract ALL columns present in the dataset and provide thorough analysis for each."""

            user_prompt = f"Analyze each column in this dataset and provide detailed profiles:\n\n{dataframe_text}"

            response : ParsedChatCompletion[ColumnProfilesSchema] = await self.structured_chat_completion(
                model=settings.INFERENCE_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format=ColumnProfilesSchema,
                max_completion_tokens=settings.INFERNECE_STRUCTURED_OUTPUTS_MAX_TOKENS,
            )
            
            if (response and 
                hasattr(response, 'choices') and 
                response.choices and 
                len(response.choices) > 0 and
                hasattr(response.choices[0], 'message') and
                hasattr(response.choices[0].message, 'parsed') and
                response.choices[0].message.parsed):
                parsed_result = response.choices[0].message.parsed
                columns = [column.model_dump() for column in parsed_result.columns]
                return columns
            else:
                logger.error("Failed to parse structured column profiles output or no valid response received")
                return []
            
        except Exception as e:
            logger.error(f"Error generating column profiles: {e}")
            return []


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
        embeddings = await model_handler.embed(test_payload)
        
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

def get_global_model_handler() -> ModelHandler:
    global _global_model_handler
    if _global_model_handler is None:
        _global_model_handler = ModelHandler()
    return _global_model_handler

async def cleanup_global_model_handler():
    global _global_model_handler
    if _global_model_handler is not None:
        await _global_model_handler.shutdown()
        _global_model_handler = None

if __name__ == "__main__":
    asyncio.run(main_test())