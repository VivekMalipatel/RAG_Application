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
        if documents and isinstance(documents[0], str):
            processed_documents = []
            for doc in documents:
                if isinstance(doc, str):
                    try:
                        parsed_doc = json.loads(doc)
                        if isinstance(parsed_doc.get("data"), str):
                            parsed_doc = {"data": [{"type": "text", "text": parsed_doc["data"]}]}
                        processed_documents.append(parsed_doc["data"])
                    except (json.JSONDecodeError, KeyError):
                        processed_documents.append([{"type": "text", "text": doc}])
                else:
                    processed_documents.append(doc)
            documents = processed_documents
        tasks = [self._aembed_single(doc) for doc in documents]
        return await asyncio.gather(*tasks)

    async def aembed_query(self, query: Union[List[dict], str]) -> List[float]:
        if isinstance(query, str):
            query = [{"type": "text", "text": query}]
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
