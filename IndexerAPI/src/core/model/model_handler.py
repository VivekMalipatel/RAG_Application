import base64
import logging
from typing import Any, Dict, List, Optional
import json
from openai import AsyncOpenAI
import aiohttp
import asyncio
from pydantic import BaseModel
from openai.resources.chat.completions.completions import ChatCompletion, ParsedChatCompletion
from core.config import settings

logger = logging.getLogger(__name__)

_global_model_handler: Optional["ModelHandler"] = None

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
    def __init__(self, api_key: str | None = None, api_base: str | None = None):
        self.embedding_api_key = api_key or settings.EMBEDDING_API_KEY
        self.embedding_api_base = api_base or settings.EMBEDDING_API_BASE
        self.inference_api_key = api_key or settings.INFERENCE_API_KEY
        self.inference_api_base = api_base or settings.INFERENCE_API_BASE
        self.embedding_client = AsyncOpenAI(
            api_key=self.embedding_api_key,
            base_url=self.embedding_api_base,
            timeout=settings.EMBEDDING_CLIENT_TIMEOUT,
            max_retries=settings.RETRIES,
        )
        self.inference_client = AsyncOpenAI(
            api_key=self.inference_api_key,
            base_url=self.inference_api_base,
            timeout=settings.EMBEDDING_CLIENT_TIMEOUT,
            max_retries=settings.RETRIES,
        )
        self._closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.shutdown()

    async def shutdown(self):
        if self._closed:
            return
        await self.embedding_client.close()
        await self.inference_client.close()
        self._closed = True

    async def generate_text_description(self, image_base64: str) -> str:
        system_prompt = """You are an AI assistant whose job is to generate rich, descriptive text for the provided documents in a multimodal RAG pipeline. For each input document-whether it comes from a PDF page, a webpage screenshot, a DOCX export, or a standalone photo—you will produce concise, context-aware text that:

        1. Identifies and names all salient Named and Unnamed entities, objects, and data visible in the document.
        2. Describes relationships, actions, or interactions depicted.
        3. Conveys any relevant context or setting needed for understanding the document.
        4. Remains clear and unambiguous, suitable for embedding alongside this document to provide downstream models with full context.

        Note: The provided image can be a screenshot of a webpage, a PDF page, or any other image format. Your task is to generate text that accurately describes the content of the document.

        Your text will be attached to each document before indexing, ensuring that the multimodal retrieval system can leverage both visual and textual cues effectively.
        """
        if not image_base64:
            raise ValueError("Empty image_base64 provided for text generation")
        response: ChatCompletion = await self.chat_completion(
            model=settings.VLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Generate alt text for the following document. Just the alt text, no other text like 'Here is the alt text:',etc",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64," + image_base64,
                            },
                        },
                    ],
                },
            ],
        )
        try:
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty text description content")
            return content.strip()
        except Exception as exc:
            logger.error(f"Failed generating text description: {exc}")
            raise

    async def embed(self, messages: List[dict]) -> List[List[float]]:
        return await self._embed_internal(messages)

    async def _embed_internal(self, messages: List[dict]) -> List[List[float]]:
        url = f"{self.embedding_api_base}/embeddings"
        headers = {
            "Authorization": "Bearer " + self.embedding_api_key,
            "Content-Type": "application/json",
        }
        settings.RETRIES = 1 if settings.RETRIES < 1 else settings.RETRIES
        for attempt in range(settings.RETRIES):
            try:
                async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=settings.EMBEDDING_CLIENT_TIMEOUT)
                ) as session:
                    async with session.post(
                        url,
                        headers=headers,
                        json={
                            "model": settings.EMBEDDING_MODEL,
                            "messages": messages,
                            "encoding_format": "float",
                        },
                    ) as response:
                        response.raise_for_status()
                        data = await response.json()
                        embeddings = [item["embedding"] for item in data["data"]]
                        return embeddings
            except Exception as exc:
                if attempt == settings.RETRIES - 1:
                    logger.error(
                        f"Error generating image embeddings after {settings.RETRIES} attempts: {exc}"
                    )
                    raise
                logger.warning(
                    f"Attempt {attempt + 1} failed for image embeddings: {exc}. Retrying..."
                )
                await asyncio.sleep(settings.RETRY_DELAY)

    async def chat_completion(self, **kwargs):
        return await self.inference_client.chat.completions.create(**kwargs)

    async def structured_chat_completion(self, **kwargs):
        return await self.inference_client.beta.chat.completions.parse(**kwargs)

    async def embed_text(self, texts: List[str]) -> List[List[float]]:
        response = await self.embedding_client.embeddings.create(
            input=texts,
            model=settings.EMBEDDING_MODEL,
            encoding_format="float",
        )
        embeddings = [item.embedding for item in response.data]
        logger.info(f"Successfully generated {len(embeddings)} text embeddings")
        return embeddings

    async def extract_entities_relationships(self, messages: List[dict]) -> Dict[str, Any]:
        extraction_prompt = """
        CRITICAL EXTRACTION REQUIREMENTS:
        1. Extract all identifiable information with high precision for general use cases
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

        DOCUMENT STRUCTURE EXTRACTION:
        - Document title, subtitle, and purpose
        - Section headings and organization
        - Document type and category
        - Version information and dates
        - Author and contributor information
        - Key topics and themes
        - Important data and statistics
        - Contact information and references

        TABULAR DATA EXTRACTION:
        - Extract structured data from tables and charts
        - Identify headers, columns, and data relationships
        - Capture numerical data and statistics
        - Extract schedules and timeline information
        - Process organizational charts and diagrams
        - Extract performance metrics and data

        GENERAL DOCUMENT EXTRACTION:
        - Main topics and themes
        - Key concepts and ideas
        - Important facts and information
        - Procedures and instructions
        - Recommendations and conclusions
        - Contact information and references
        - Dates and temporal information
        - Quantitative data and measurements

        The goal is to provide comprehensive document intelligence with entities (Maximum of 50 top most entities) and relationships extraction for any general or personal context, capturing every significant detail relevant for understanding, analysis, and knowledge management.
        
        Output Schema:

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
        """
        system_message = {"role": "system", "content": extraction_prompt}
        has_image = len(messages[0]["content"]) > 1
        if has_image:
            image_messages = [
                {
                    "role": "user",
                    "content": [messages[0]["content"][0]],
                }
            ]
            text_messages = [
                {
                    "role": "user",
                    "content": [messages[0]["content"][1]],
                }
            ]
        else:
            image_messages = []
            text_messages = [
                {
                    "role": "user",
                    "content": [messages[0]["content"][0]],
                }
            ]
        retries = max(1, settings.RETRIES)

        if has_image:
            for attempt in range(retries):
                try:
                    response = await self.structured_chat_completion(
                        model=settings.VLM_MODEL,
                        messages=[system_message] + image_messages,
                        response_format=EntityRelationSchema,
                        max_completion_tokens=settings.VLM_MAX_TOKENS,
                    )
                    parsed = response.choices[0].message.parsed
                    entities = [entity.model_dump() for entity in parsed.entities]
                    relationships = [rel.model_dump() for rel in parsed.relationships]
                    return {"entities": entities, "relationships": relationships}
                except Exception as exc:
                    if attempt == retries - 1:
                        logger.error(
                            f"Image-based entity extraction failed after {retries} attempts: {exc}"
                        )
                        raise
                    logger.warning(
                        f"Attempt {attempt + 1}: image-based entity extraction failed: {exc}. Retrying..."
                    )
                if attempt != retries - 1:
                    await asyncio.sleep(settings.RETRY_DELAY)

        response = await self.structured_chat_completion(
            model=settings.REASONING_MODEL,
            messages=[system_message] + text_messages,
            response_format=EntityRelationSchema,
            max_completion_tokens=settings.REASONING_MAX_TOKENS,
        )
        try:
            parsed = response.choices[0].message.parsed
            entities = [entity.model_dump() for entity in parsed.entities]
            relationships = [rel.model_dump() for rel in parsed.relationships]
            return {"entities": entities, "relationships": relationships}
        except Exception as exc:
            logger.error(f"Failed to extract entities via text pathway: {exc}")
            raise

    async def embed_entity_relationship_profiles(
        self,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        entity_profiles = [entity.get("entity_profile", "") for entity in entities]
        relationship_profiles = [rel.get("relation_profile", "") for rel in relationships]
        all_profiles = entity_profiles + relationship_profiles
        if not all_profiles:
            return entities, relationships
        embeddings = await self.embed_text(all_profiles)
        entity_embeddings = embeddings[: len(entity_profiles)]
        relationship_embeddings = embeddings[len(entity_profiles) :]
        for index, entity in enumerate(entities):
            if index < len(entity_embeddings):
                entity["embedding"] = entity_embeddings[index]
        for index, rel in enumerate(relationships):
            if index < len(relationship_embeddings):
                rel["embedding"] = relationship_embeddings[index]
        return entities, relationships

    async def generate_structured_summary(self, dataframe_text: str) -> str:
        system_prompt = """You are an AI assistant that analyzes structured data and provides comprehensive summaries. 
        Given a tabular dataset representation, provide a clear, informative summary that includes:
        1. Overall purpose and content of the dataset
        2. Key patterns and insights
        3. Data quality observations
        4. Notable trends or anomalies
        5. Potential use cases or applications
        
        Keep the summary concise but comprehensive, suitable for embedding and retrieval."""
        user_prompt = (
            "Analyze this structured dataset and provide a comprehensive summary:\n\n"
            f"{dataframe_text[: settings.REASONING_MAX_TOKENS - 10000]}"
        )
        response: ChatCompletion = await self.chat_completion(
            model=settings.REASONING_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_completion_tokens=settings.REASONING_MAX_TOKENS,
            temperature=0.5,
            top_p=0.2,
            frequency_penalty=0.5,
        )
        try:
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty structured summary content")
            return content.strip()
        except Exception as exc:
            logger.error(f"Failed to generate structured summary: {exc}")
            raise

    async def generate_column_profiles(self, dataframe_text: str) -> List[Dict[str, str]]:
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
        user_prompt = (
            "Analyze each column in this dataset and provide detailed profiles:\n\n"
            f"{dataframe_text[: settings.REASONING_MAX_TOKENS - 10000]}"
        )
        response: ParsedChatCompletion[
            ColumnProfilesSchema
        ] = await self.structured_chat_completion(
            model=settings.REASONING_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=ColumnProfilesSchema,
            max_completion_tokens=settings.REASONING_MAX_TOKENS,
        )
        try:
            parsed_result = response.choices[0].message.parsed
            columns = [column.model_dump() for column in parsed_result.columns]
            if not columns:
                raise ValueError("Empty column profiles content")
            return columns
        except Exception as exc:
            logger.error(f"Failed to generate column profiles: {exc}")
            raise

async def main_test():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    test_payload = [
        {
            "image": "iVBORw0KGgoAAAANSUhEUgAAAIAAAACACAIAAABMXPacAAABLUlEQVR4nO3RQREAIAzAMMC/501GHjQKetc7MyfO0wG/awDWAKwBWAOwBmANwBqANQBrANYArAFYA7AGYA3AGoA1AGsA1gCsAVgDsAZgDcAagDUAawDWAKwBWAOwBmANwBqANQBrANYArAFYA7AGYA3AGoA1AGsA1gCsAVgDsAZgDcAagDUAawDWAKwBWAOwBmANwBqANQBrANYArAFYA7AGYA3AGoA1AGsA1gCsAVgDsAZgDcAagDUAawBbgZgP9Ag5IZgAAAABJRU5ErkJggg==",
            "text": "This is a sample image",
        }
    ]
    try:
        model_handler = ModelHandler()
        embeddings = await model_handler.embed(test_payload)
        if embeddings:
            logger.info(
                "Generated %s embeddings with first dimension length %s",
                len(embeddings),
                len(embeddings[0]),
            )
        text_embeddings = await model_handler.embed_text(["This is a sample text for embedding."])
        if text_embeddings:
            logger.info(
                "Generated text embedding with dimension length %s",
                len(text_embeddings[0]),
            )
    except Exception as exc:
        logger.error(f"Error during testing: {str(exc)}")

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
