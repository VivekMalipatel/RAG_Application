"""
Text generation handler for Model Router API.

This module provides utilities for generating text using
the Model Router API's LLM capabilities.
"""

import logging
from typing import Dict, Any, List, Optional, Union, Callable
import json

from app.core.model.api_client import ModelClient

logger = logging.getLogger(__name__)

class TextGenerator:
    """Text generation handler for interacting with LLMs via Model Router API."""
    
    def __init__(
        self, 
        model_client: Optional[ModelClient] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ):
        """
        Initialize the Text Generator.
        
        Args:
            model_client: Initialized ModelClient instance
            model: Model identifier to use for generation
            temperature: Sampling temperature (higher = more random)
            max_tokens: Maximum number of tokens to generate
        """
        self.client = model_client or ModelClient()
        self.model = model or self.client.config.default_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        logger.info(f"Initialized Text Generator with model: {self.model}")
    
    async def generate_text(
        self, 
        prompt: str, 
        system_message: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Generate text using the Model Router API.
        
        Args:
            prompt: User prompt/query to send to the model
            system_message: Optional system message to set context
            temperature: Sampling temperature (higher = more random)
            max_tokens: Maximum number of tokens to generate
            stream: Whether to stream the response
            
        Returns:
            Dictionary containing the response content and metadata
        """
        try:
            messages = []
            
            # Add system message if provided
            if system_message:
                messages.append({"role": "system", "content": system_message})
                
            # Add user message (the prompt)
            messages.append({"role": "user", "content": prompt})
            
            # Use instance defaults if specific values aren't provided
            temp = temperature if temperature is not None else self.temperature
            tokens = max_tokens if max_tokens is not None else self.max_tokens
            
            logger.info(f"Generating text with prompt: {prompt[:50]}... [model: {self.model}]")
            
            # Call the Model Router API
            response = self.client.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temp,
                max_tokens=tokens,
                stream=stream
            )
            
            if stream:
                # Return the streaming response for the caller to process
                return {"success": True, "stream": response}
            else:
                # Extract and return the response content
                content = response.choices[0].message.content
                logger.debug(f"Generated content: {content[:100]}...")
                
                return {
                    "success": True,
                    "content": content,
                    "model": self.model,
                    "finish_reason": response.choices[0].finish_reason,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    }
                }
                
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "model": self.model
            }
    
    async def process_streaming_response(
        self, 
        response_stream, 
        callback: Optional[Callable[[str], None]] = None
    ) -> str:
        """
        Process a streaming response from the text generation API.
        
        Args:
            response_stream: Streaming response from generate_text
            callback: Optional callback function to process each chunk
            
        Returns:
            Complete generated text after streaming completes
        """
        full_text = ""
        
        try:
            for chunk in response_stream:
                if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                    content = chunk.choices[0].delta.content
                    if content:
                        full_text += content
                        if callback:
                            callback(content)
                            
            return full_text
        except Exception as e:
            logger.error(f"Error processing streaming response: {str(e)}")
            return full_text
    
    async def generate_structured_output(
        self, 
        prompt: str, 
        output_schema: Dict[str, Any],
        system_message: Optional[str] = None,
        temperature: Optional[float] = 0.2,  # Lower temperature for structured outputs
    ) -> Dict[str, Any]:
        """
        Generate structured JSON output using Model Router API's function calling.
        
        Args:
            prompt: User prompt/query to send to the model
            output_schema: JSON Schema defining the structure of the output
            system_message: Optional system message to set context
            temperature: Sampling temperature (lower recommended for structured data)
            
        Returns:
            Dictionary containing the structured response data
        """
        try:
            messages = []
            
            # Add system message if provided
            if system_message:
                messages.append({"role": "system", "content": system_message})
            else:
                # Default system message for structured output
                messages.append({
                    "role": "system", 
                    "content": "You are a helpful assistant that outputs structured data according to the provided schema."
                })
                
            # Add user message (the prompt)
            messages.append({"role": "user", "content": prompt})
            
            # Define the function for the model to call
            function_name = "generate_structured_output"
            functions = [
                {
                    "name": function_name,
                    "description": "Generate structured data based on the user query",
                    "parameters": output_schema
                }
            ]
            
            logger.info(f"Generating structured output for prompt: {prompt[:50]}...")
            
            # Call the Model Router API with function calling
            response = self.client.client.chat.completions.create(
                model=self.model,
                messages=messages,
                functions=functions,
                function_call={"name": function_name},
                temperature=temperature if temperature is not None else 0.2
            )
            
            # Extract the function call response
            function_args = response.choices[0].message.function_call.arguments
            
            # Parse the JSON string into a Python dictionary
            structured_data = json.loads(function_args)
            
            return {
                "success": True,
                "data": structured_data,
                "model": self.model,
            }
                
        except Exception as e:
            logger.error(f"Error generating structured output: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "model": self.model
            }