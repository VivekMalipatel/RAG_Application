import logging
from app.services.upload_file.upload_request_validator import RequestValidator

class UploadRequestReceiver:
    """Handles incoming file upload requests and forwards them for validation."""

    def __init__(self,minio_config: dict, db_url: str, redis_url: str):
        self.validator = RequestValidator(minio_config, db_url, redis_url)

    async def receive_upload_request(self, request_data: dict, file_data: bytes = None):

        try:
            if not request_data.get("user_id"):
                logging.error("Unauthorized request: Missing user_id.")
                return {"success": False, "error": "Unauthorized request."}

            logging.info(f"Received upload request from user: {request_data['user_id']}, File: {request_data['file_name']}")

            validation_response = await self.validator.validate_request(request_data, file_data)

            logging.debug(f"Validation response: {validation_response}")
            
            return validation_response

        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            return {"success": False, "error": "Internal server error."}