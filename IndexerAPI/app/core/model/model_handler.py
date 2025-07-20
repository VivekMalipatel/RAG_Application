import base64
import logging
from typing import List, Dict, Any, Optional
import json
from openai import AsyncOpenAI
import httpx
import asyncio
from pydantic import BaseModel
from openai.resources.chat.completions.completions import ChatCompletion, ParsedChatCompletion
import requests
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
        http_client_embedding = httpx.AsyncClient(timeout=settings.EMBEDDING_CLIENT_TIMEOUT)
        http_client = httpx.AsyncClient(timeout=settings.INFERENCE_CLIENT_TIMEOUT)
        self.embedding_client = AsyncOpenAI(api_key=self.embedding_api_key, base_url=self.embedding_api_base, http_client=http_client_embedding)
        self.inference_client = AsyncOpenAI(api_key=self.inference_api_key, base_url=self.inference_api_base, http_client=http_client)
        
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
        
        max_retries = 3
        for attempt in range(max_retries):
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
                alt_text = response.choices[0].message.content.strip()
                return alt_text
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Error generating alt text after {max_retries} attempts: {str(e)}")
                    raise
                else:
                    logger.warning(f"Attempt {attempt + 1} failed for text description: {str(e)}. Retrying...")
                    await asyncio.sleep(2)

    async def embed(self, messages: List[dict]) -> List[List[float]]:
        self._start_queue_processors()
        return await self.embedding_rate_limiter.add_request(messages)

    async def _embed_internal(self, messages: List[dict]) -> List[List[float]]:

        url = f"{self.embedding_api_base}/embeddings"
        headers = {
            "Authorization": "Bearer " + self.embedding_api_key,
            "Content-Type": "application/json"
        }
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=settings.EMBEDDING_CLIENT_TIMEOUT) as client:
                    response = await client.post(
                        url,
                        headers=headers,
                        json={
                            "model": settings.EMBEDDING_MODEL,
                            "messages": messages,
                            "encoding_format": "float"
                        }
                    )
                    response.raise_for_status()
                    data = response.json()
                    embeddings = [item["embedding"] for item in data["data"]]
                    return embeddings
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Error generating image embeddings after {max_retries} attempts: {str(e)}")
                    raise
                else:
                    logger.warning(f"Attempt {attempt + 1} failed for image embeddings: {str(e)}. Retrying...")
                    await asyncio.sleep(5)
    
    async def _chat_completion_internal(self, **kwargs):
        try:
            response = await self.inference_client.chat.completions.create(**kwargs)
            return response
        except Exception as e:
            logger.error(f"Error in chat completion: {str(e)}")
            raise
    
    async def _structured_chat_completion_internal(self, **kwargs):
        try:
            response = await self.inference_client.beta.chat.completions.parse(**kwargs)
            return response
        except Exception as e:
            logger.error(f"Error in structured chat completion: {str(e)}")
            raise
    
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
            raise

    def encode_image_to_base64(self, image_path: str) -> str:
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encoding image to base64: {str(e)}")
            raise

    async def extract_entities_relationships(self, messages: List[dict]) -> Dict[str, Any]:
        try:
            extraction_prompt = """Extract comprehensive entities, relationships, and document metadata from the given content for general and personal document analysis.

            CRITICAL EXTRACTION REQUIREMENTS:
            1. Extract ALL identifiable information with high precision for general use cases
            2. Create entity IDs using lowercase with underscores: "John Smith" -> "john_smith"
            3. Extract relationships with contextual information
            4. Provide detailed entity profiles with roles, descriptions, and contextual standing
            5. Handle coreference resolution (he/she -> actual name)
            6. Extract document structure and content elements
            7. For images, consider visual entities, charts, diagrams, and relationships

            JSON SCHEMA DESCRIPTION:
            Your response must be a JSON object containing two main arrays:
            - "entities": An array of entity objects, each representing a distinct person, place, thing, concept, or identifier found in the content
            - "relationships": An array of relationship objects, each describing how two entities are connected or related
            
            Each entity object must contain:
            - "id": A unique identifier in lowercase with underscores (e.g., "john_smith", "acme_corporation")
            - "text": The original text as it appears in the document
            - "entity_type": The category of entity from the predefined types
            - "entity_profile": A detailed description of the entity's role, context, and significance
            
            Each relationship object must contain:
            - "source": The ID of the first entity in the relationship
            - "target": The ID of the second entity in the relationship
            - "relation_type": The type of relationship from the predefined types
            - "relation_profile": A detailed description of how and why these entities are related

            output format:

            {
                "entities": [
                    {
                        "id": "entity_id_lowercase_with_underscores",
                        "text": "Original entity text",
                        "entity_type": "ENTITY_TYPE",
                        "entity_profile": "Detailed profile description"
                    }
                ],
                "relationships": [
                    {
                        "source": "source_entity_id",
                        "target": "target_entity_id",
                        "relation_type": "RELATIONSHIP_TYPE",
                        "relation_profile": "Detailed relationship description"
                    }
                ]
            }
            

            COMPREHENSIVE ENTITY TYPES:
            - PERSON: Individuals, authors, speakers, contacts, stakeholders, customers, family members
            - ORGANIZATION: Companies, institutions, groups, teams, departments, agencies
            - LOCATION: Places, addresses, cities, countries, regions, facilities, venues
            - DOCUMENT: Files, reports, articles, books, papers, presentations, manuals
            - IDENTIFIER: IDs, codes, numbers, references, accounts, serial numbers
            - CONCEPT: Ideas, topics, subjects, themes, methodologies, principles
            - FINANCIAL: Money, costs, prices, budgets, investments, transactions
            - DATE_TIME: Dates, times, deadlines, schedules, periods, durations
            - REQUIREMENT: Goals, objectives, specifications, criteria, standards
            - POSITION_TITLE: Job titles, roles, positions, responsibilities
            - CONTACT_INFO: Phone numbers, emails, addresses, social media
            - ASSET: Equipment, tools, resources, technology, systems
            - PROCESS: Procedures, workflows, methods, operations, activities
            - CLASSIFICATION: Categories, types, levels, priorities, statuses
            - PRODUCT_SERVICE: Products, services, offerings, solutions
            - METRIC: Measurements, statistics, performance indicators, benchmarks

            COMPREHENSIVE RELATIONSHIP TYPES:
            - WORKS_FOR: Employment or service relationship
            - MANAGES: Management or supervisory relationship
            - REPORTS_TO: Hierarchical reporting structure
            - COLLABORATES_WITH: Working partnership or cooperation
            - ASSOCIATED_WITH: General association or connection
            - LOCATED_AT: Physical or logical location
            - VALID_FROM/UNTIL: Temporal validity and duration
            - RESPONSIBLE_FOR: Accountability and ownership
            - AUTHORED_BY: Creation or authorship
            - REFERENCES: Citations, mentions, cross-references
            - CONTAINS: Structural containment or inclusion
            - PARTICIPATES_IN: Involvement or engagement
            - RELATED_TO: General relationship or connection
            - DEPENDS_ON: Dependencies and prerequisites
            - ASSIGNED_TO: Task or responsibility assignment
            - DESCRIBES: Descriptive or explanatory relationship
            - BELONGS_TO: Ownership or membership
            - COMMUNICATES_WITH: Communication or interaction

            DOCUMENT STRUCTURE Considerations:
            - Document title, subtitle, and purpose
            - Section headings and organization
            - Document type and category
            - Version information and dates
            - Author and contributor information
            - Key topics and themes
            - Important data and statistics
            - Contact information and references

            TABULAR DATA Considerations:
            - Extract structured data from tables and charts
            - Identify headers, columns, and data relationships
            - Capture numerical data and statistics
            - Extract schedules and timeline information
            - Process organizational charts and diagrams
            - Extract performance metrics and data

            GENERAL DOCUMENT Considerations:
            - Main topics and themes
            - Key concepts and ideas
            - Important facts and information
            - Procedures and instructions
            - Recommendations and conclusions
            - Contact information and references
            - Dates and temporal information
            - Quantitative data and measurements

            The goal is to provide comprehensive document intelligence with entities and relationships extraction for any general or personal context, capturing every significant detail relevant for understanding, analysis, and knowledge management.            
            """

            if messages is not None:
                messages = [{"role": "system", "content": extraction_prompt}] + messages
            else:
                raise ValueError("Either messages or text must be provided")

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    structured_response : ParsedChatCompletion[EntityRelationSchema] = await self.structured_chat_completion(
                        model=settings.INFERENCE_MODEL,
                        messages=messages,
                        response_format=EntityRelationSchema,
                        max_completion_tokens=30000
                    )
                    
                    if structured_response.choices[0].message.parsed:
                        parsed_result = structured_response.choices[0].message.parsed
                        entities = [entity.model_dump() for entity in parsed_result.entities]
                        relationships = [rel.model_dump() for rel in parsed_result.relationships]
                        
                        for entity in entities:
                            entity["id"] = entity["id"].lower().replace(" ", "_").replace("-", "_")
                        
                        for rel in relationships:
                            rel["source"] = rel["source"].lower().replace(" ", "_").replace("-", "_")
                            rel["target"] = rel["target"].lower().replace(" ", "_").replace("-", "_")
                        
                        return {"entities": entities, "relationships": relationships}
                        
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Entity extraction failed after {max_retries} attempts: {e}")
                        return {"entities": [], "relationships": []}
                    else:
                        logger.warning(f"Attempt {attempt + 1} failed for entity extraction: {str(e)}. Retrying...")
                        await asyncio.sleep(5)
            
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
            
            return response.choices[0].message.content.strip()
            
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
            )
            
            if response.choices[0].message.parsed:
                parsed_result = response.choices[0].message.parsed
                columns = [column.model_dump() for column in parsed_result.columns]
                return columns
            else:
                logger.error("Failed to parse structured column profiles output")
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