import logging
import os
import json
from typing import Dict, Any, List, Optional, Union
import asyncio

from app.core.model.api_client import ModelClient

logger = logging.getLogger(__name__)

class TextGenerator:
    
    def __init__(
        self,
        model: Optional[str] = None,
        api_client: Optional[ModelClient] = None
    ):
        self.model = model or os.environ.get("DEFAULT_TEXT_MODEL", "gpt-3.5-turbo")
        self.api_client = api_client or ModelClient()
        logger.info(f"Initialized Text Generator with model: {self.model}")
    
    async def generate_text(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        logger.info(f"Generating text with model: {model or self.model}")
        
        try:
            messages = []
            
            if system_message:
                messages.append({"role": "system", "content": system_message})
                
            messages.append({"role": "user", "content": prompt})
            
            data = {
                "model": model or self.model,
                "messages": messages,
                "temperature": temperature
            }
            
            if max_tokens:
                data["max_tokens"] = max_tokens
                
            response = await self.api_client._make_request(
                "POST",
                "/chat/completions",
                data=data,
                stream=stream
            )
            
            if stream:
                return response
            
            generated_text = response["choices"][0]["message"]["content"]
            usage = response.get("usage", {})
            
            return {
                "success": True,
                "content": generated_text,
                "usage": usage,
                "model": model or self.model
            }
            
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def generate_structured_output(
        self,
        prompt: str,
        output_schema: Dict[str, Any],
        system_message: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        logger.info(f"Generating structured output with model: {model or self.model}")
        
        try:
            messages = []
            
            if system_message:
                messages.append({"role": "system", "content": system_message})
                
            messages.append({"role": "user", "content": prompt})
            
            functions = [
                {
                    "name": "generate_structured_output",
                    "description": "Generate a structured output based on the user's query",
                    "parameters": output_schema
                }
            ]
            
            data = {
                "model": model or self.model,
                "messages": messages,
                "functions": functions,
                "function_call": {"name": "generate_structured_output"},
                "temperature": temperature
            }
            
            response = await self.api_client._make_request(
                "POST",
                "/chat/completions",
                data=data
            )
            
            function_call = response["choices"][0]["message"].get("function_call", {})
            
            if not function_call or "arguments" not in function_call:
                logger.error("No function call in response")
                return {
                    "success": False,
                    "error": "Failed to generate structured output"
                }
                
            try:
                structured_data = json.loads(function_call["arguments"])
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing function arguments: {str(e)}")
                return {
                    "success": False,
                    "error": f"Invalid JSON in function arguments: {str(e)}"
                }
                
            usage = response.get("usage", {})
            
            return {
                "success": True,
                "data": structured_data,
                "usage": usage,
                "model": model or self.model
            }
            
        except Exception as e:
            logger.error(f"Error generating structured output: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }