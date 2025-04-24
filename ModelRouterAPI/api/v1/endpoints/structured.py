import logging
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException
from pydantic import ValidationError

from model_handler import ModelRouter
from model_type import ModelType
from schemas.structured import StructuredOutputRequest, StructuredOutputResponse
from core.security import get_api_key

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/generate", response_model=StructuredOutputResponse)
async def generate_structured_output(
    request: StructuredOutputRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Generate structured output from a model according to a provided JSON schema.
    """
    try:
        # Initialize the ModelRouter
        model_router = ModelRouter.initialize_from_model_name(
            model_name=request.model,
            model_type=ModelType.TEXT_GENERATION,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            system_prompt=request.system_prompt
        )
        
        # Generate structured output using the JSON schema provided in the request
        structured_data = await model_router.generate_structured_output(
            prompt=request.prompt,
            schema=request.schema,  # Use the JSON schema directly
            max_tokens=request.max_tokens
        )
        
        # Check if there's an error in the response
        if isinstance(structured_data, dict) and "error" in structured_data:
            raise HTTPException(
                status_code=422, 
                detail=f"Failed to generate valid structured output: {structured_data.get('error')}"
            )
            
        # Create response
        response = StructuredOutputResponse(
            model=request.model,
            output=structured_data,
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None
        )
        
        return response
        
    except ValidationError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=422, detail=f"Invalid request: {str(e)}")
    except ValueError as e:
        logger.error(f"Value error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except NotImplementedError as e:
        logger.error(f"Not implemented: {str(e)}")
        raise HTTPException(status_code=501, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")