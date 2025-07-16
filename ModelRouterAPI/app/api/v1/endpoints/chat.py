from typing import Dict, Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse, JSONResponse

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

@router.post("/chat/completions", response_model=None)
async def create_chat_completion(
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
        
        try:
            model_router = await ModelRouter.initialize_from_model_name(
                model_name=request_data.get("model"),
                model_type=ModelType.TEXT_GENERATION
            )
        except Exception as model_error:
            return create_error_response(
                f"The model '{request_data.get('model')}' does not exist or is not available",
                "model_not_found",
                404
            )
        
        try:
            if request_data.get("stream", False):
                return StreamingResponse(
                    generate_chat_stream(request_data, model_router),
                    media_type="text/event-stream"
                )
            else:
                response = await model_router.generate_text(**request_data)
                
                return response
                
        except Exception as e:
            return create_error_response(
                f"Error generating chat completion: {str(e)}",
                "chat_completion_error",
                500
            )
    
    except Exception as e:
        return create_error_response(
            str(e),
            "internal_server_error",
            500
        )

async def generate_chat_stream(
    request_data: Dict[str, Any],
    model_router: ModelRouter,
):
    try:
        stream = await model_router.generate_text(**request_data)
        
        async for chunk in stream:
            
            if isinstance(chunk, str):
                yield f"data: {chunk}\n\n"
            else:
                import json
                yield f"data: {json.dumps(chunk)}\n\n"
        
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        import json
        error_chunk = {
            "error": {
                "message": f"Error in streaming: {str(e)}",
                "type": "streaming_error"
            }
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
        yield "data: [DONE]\n\n"