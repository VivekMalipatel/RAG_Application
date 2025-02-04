import logging
from app.services.upload_request_validator import RequestValidator

class UploadRequestReceiver:
    """Handles incoming file upload requests and forwards them for validation."""

    def __init__(self):
        self.validator = RequestValidator()  # Instance of RequestValidator

    async def receive_upload_request(self, request_data: dict):
        """
        Processes an incoming file upload request.

        Args:
            request_data (dict): Contains file details such as:
                - user_id (str)
                - file_name (str)
                - relative_path (str)
                - upload_id (str, optional for multipart)
                - chunk_number (int, optional for multipart)
                - file_size (int)
                - mime_type (str)
                - metadata (dict, optional)

        Returns:
            dict: Response indicating whether the request is valid.
        """
        try:
            # Extract user_id for authentication
            user_id = request_data.get("user_id")
            if not user_id:
                logging.error("Unauthorized request: Missing user_id.")
                return {"success": False, "error": "Unauthorized request."}

            logging.info(f"Received upload request from user: {user_id}, File: {request_data['file_name']}")

            # Forward request to Request Validator (ensure it's awaited)
            validation_response = await self.validator.validate_request(request_data)

            return validation_response

        except Exception as e:
            logging.error(f"Unexpected error in upload request processing: {e}")
            return {"success": False, "error": "Internal server error."}