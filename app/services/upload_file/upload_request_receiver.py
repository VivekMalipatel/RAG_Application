import logging
from app.services.upload_file.upload_request_validator import RequestValidator

class UploadRequestReceiver:
    """Handles incoming file upload requests and forwards them for validation."""

    def __init__(self):
        self.validator = RequestValidator()

    async def receive_upload_request(self, request_data: dict, file_data: bytes = None):
        """
        Processes an incoming file upload request.

        Args:
            request_data (dict): User and file details.
            file_data (bytes): Optional chunk data.
        """
        try:
            if not request_data.get("user_id"):
                logging.error("Unauthorized request: Missing user_id.")
                return {"success": False, "error": "Unauthorized request."}

            logging.info(f"Received upload request from user: {request_data['user_id']}, File: {request_data['file_name']}")

            validation_response = await self.validator.validate_request(request_data, file_data)

            return validation_response

        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            return {"success": False, "error": "Internal server error."}