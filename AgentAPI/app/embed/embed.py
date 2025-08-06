import asyncio
import aiohttp
import logging
import json
from typing import List, Union
from langchain_openai import OpenAIEmbeddings
from config import config

class JinaEmbeddings(OpenAIEmbeddings):
    def __init__(self, **kwargs):
        kwargs.setdefault('base_url', config.OPENAI_BASE_URL)
        kwargs.setdefault('api_key', config.OPENAI_API_KEY)
        kwargs.setdefault('model', config.MULTIMODEL_EMBEDDING_MODEL)
        super().__init__(**kwargs)
        self._logger = logging.getLogger(__name__)

    async def aembed_documents(self, documents: List[List[dict]]) -> List[List[float]]:
        self._logger.debug(f"aembed_documents called with: {documents}")
        
        if documents and isinstance(documents[0], str):
            processed_documents = await asyncio.gather(*[
                self._process_single_document(doc) for doc in documents
            ])
        else:
            processed_documents = documents
        
        self._logger.debug(f"Processed documents: {processed_documents}")
        
        embedding_tasks = [
            self._get_embedding_or_zero(doc) for doc in processed_documents
        ]
        
        results = await asyncio.gather(*embedding_tasks)
        return results

    async def _process_single_document(self, doc):
        if not isinstance(doc, str):
            return doc
            
        try:
            parsed_doc = json.loads(doc)
            if not parsed_doc or parsed_doc == [] or parsed_doc == "[]":
                self._logger.debug(f"Processing empty document: {doc}")
                return None
            elif isinstance(parsed_doc.get("data"), str):
                parsed_doc = {"data": [{"type": "text", "text": parsed_doc["data"]}]}
                return parsed_doc["data"]
            elif isinstance(parsed_doc, list):
                if not parsed_doc:
                    self._logger.debug(f"Processing empty list document: {doc}")
                    return None
                else:
                    return parsed_doc
            else:
                return parsed_doc["data"]
        except (json.JSONDecodeError, KeyError):
            if doc == "[]" or doc == "" or not doc.strip():
                self._logger.debug(f"Processing invalid/empty JSON document: {doc}")
                return None
            else:
                return [{"type": "text", "text": doc}]

    async def _get_embedding_or_zero(self, doc):
        if doc is None:
            self._logger.debug("Skipping API call for empty document, using zero vector")
            return [0.0] * config.MULTIMODEL_EMBEDDING_MODEL_DIMS
        else:
            return await self._aembed_single(doc)

    async def aembed_query(self, query: Union[List[dict], str]) -> List[float]:
        if isinstance(query, str):
            if not query or not query.strip():
                self._logger.debug("Skipping API call for empty query, using zero vector")
                return [0.0] * config.MULTIMODEL_EMBEDDING_MODEL_DIMS
            query = [{"type": "text", "text": query}]
        elif not query:
            self._logger.debug("Skipping API call for empty query, using zero vector")
            return [0.0] * config.MULTIMODEL_EMBEDDING_MODEL_DIMS
        return await self._aembed_single(query)

    async def _aembed_single(self, content: List[dict]) -> List[float]:
        url = f"{self.openai_api_base}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.openai_api_key._secret_value}",
            "Content-Type": "application/json"
        }
        
        messages = [{"role": "user", "content": content}]
        retries = max(1, config.EMBEDDING_CLEINT_RETRIES)
        for attempt in range(retries):
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=config.EMBEDDING_CLIENT_TIMEOUT)) as session:
                    async with session.post(
                        url,
                        headers=headers,
                        json={
                            "model": self.model,
                            "messages": messages,
                            "encoding_format": "float"
                        }
                    ) as response:
                        response.raise_for_status()
                        self._logger.info(f"HTTP Request: POST {url} \"HTTP/1.1 {response.status} {response.reason}\"")
                        data = await response.json()
                        embeddings = [item["embedding"] for item in data["data"]]
                        if embeddings:
                            return embeddings[0]
            except Exception as e:
                self._logger.error(f"Embedding request attempt {attempt+1} failed: {e}")
                if attempt == retries - 1:
                    raise
        raise ValueError("Failed to generate embedding after all retries")
