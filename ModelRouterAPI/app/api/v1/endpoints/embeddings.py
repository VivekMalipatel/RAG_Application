from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from core.model_handler import ModelRouter
from core.model_type import ModelType

router = APIRouter()

def create_error_response(message: str, code: str, status_code: int = 400):
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "param": None,
                "code": code
            }
        }
    )

@router.post("")
async def create_embeddings(
    request: Request,
):
    
    try:
        request_data = await request.json()
        
        if not request_data.get("model"):
            return create_error_response(
                "Model parameter is required",
                "invalid_request_error", 
                400
            )
        
        input_data = request_data.get("input")
        if input_data is None:
            input_data = request_data.get("messages")
        if isinstance(input_data, list):
            is_image_input = all(isinstance(item, dict) for item in input_data) if input_data else False
        else:
            is_image_input = isinstance(input_data, dict)
        
        try:
            if is_image_input:
                model_router = await ModelRouter.initialize_from_model_name(
                    model_name=request_data.get("model"),
                    model_type=ModelType.IMAGE_EMBEDDING
                )
            else:
                model_router = await ModelRouter.initialize_from_model_name(
                    model_name=request_data.get("model"),
                    model_type=ModelType.TEXT_EMBEDDING
                )
        except Exception as model_error:
            return create_error_response(
                f"The model '{request_data.get('model')}' does not exist or is not available",
                "model_not_found",
                404
            )
        
        try:
            response = await model_router.embed(**request_data)
            
            return response
            
        except Exception as e:
            return create_error_response(
                f"Error generating embeddings: {str(e)}",
                "embedding_error",
                500
            )
    
    except Exception as e:
        return create_error_response(
            str(e),
            "internal_server_error",
            500
        )
