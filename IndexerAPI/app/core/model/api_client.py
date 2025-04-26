import logging
import os
import json
import asyncio
from typing import Dict, Any, List, Optional, Union, Tuple
from urllib.parse import urljoin

import httpx

logger = logging.getLogger(__name__)

class ModelClient:
    
    def __init__(
        self,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 3,
        retry_delay: int = 1
    ):
        self.api_base = api_base or os.environ.get("MODEL_API_BASE", "http://localhost:8000/v1")
        self.api_key = api_key or os.environ.get("MODEL_API_KEY", "")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        logger.info(f"Initialized Model client with API base: {self.api_base}")
    
    async def get_models(self) -> Dict[str, Any]:
        logger.info("Getting available models")
        try:
            endpoint = "/models"
            response = await self._make_request("GET", endpoint)
            return response
        except Exception as e:
            logger.error(f"Error getting models: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def check_health(self) -> Dict[str, Any]:
        logger.info("Checking API health")
        try:
            endpoint = "/health"
            response = await self._make_request("GET", endpoint)
            return {"success": True, "status": response}
        except Exception as e:
            logger.error(f"API health check failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _make_request(
        self, 
        method: str, 
        endpoint: str, 
        data: Optional[Dict[str, Any]] = None, 
        params: Optional[Dict[str, Any]] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        url = urljoin(self.api_base, endpoint)
        headers = {
            "Content-Type": "application/json"
        }
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            
        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    if stream:
                        response = await client.stream(
                            method,
                            url,
                            json=data,
                            params=params,
                            headers=headers
                        )
                        return response
                    else:
                        if method.upper() == "GET":
                            response = await client.get(url, params=params, headers=headers)
                        elif method.upper() == "POST":
                            response = await client.post(url, json=data, headers=headers)
                        else:
                            raise ValueError(f"Unsupported HTTP method: {method}")
                        
                        response.raise_for_status()
                        
                        return response.json()
            
            except httpx.HTTPStatusError as e:
                logger.warning(f"HTTP error on attempt {attempt+1}/{self.max_retries}: {str(e)}")
                if attempt == self.max_retries - 1:
                    logger.error(f"Maximum retries reached. Last error: {str(e)}")
                    raise Exception(f"API request failed after {self.max_retries} attempts: {str(e)}")
                await asyncio.sleep(self.retry_delay)
                
            except Exception as e:
                logger.error(f"Request error on attempt {attempt+1}/{self.max_retries}: {str(e)}")
                if attempt == self.max_retries - 1:
                    raise Exception(f"API request failed after {self.max_retries} attempts: {str(e)}")
                await asyncio.sleep(self.retry_delay)