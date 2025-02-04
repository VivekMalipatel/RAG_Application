import logging
from app.core.validation.filevalidator import FileValidator
from app.core.kafka.kafka_handler import KafkaHandler

class FileUploadReceiver:
    """
    Handles the initial upload request validation before sending it to Kafka for processing.
    """

    def __init__(self, kafka_topic="File_Upload_Requests"):
        """
        Initializes the file upload receiver.

        Args:
            kafka_topic (str): The Kafka topic where upload requests are sent.
        """
        self.file_validator = FileValidator()
        self.kafka_handler = KafkaHandler()
        self.kafka_topic = kafka_topic

    async def receive_upload_request(self, user_id: str, file_name: str, file_size: int, file_type: str, file_hash: str):
        """
        Receives an upload request and validates it.

        Args:
            user_id (str): ID of the user uploading the file.
            file_name (str): Name of the file.
            file_size (int): Size of the file in bytes.
            file_type (str): File MIME type.
            file_hash (str): Unique hash of the file.

        Returns:
            dict: Success or error response.
        """
        try:
            if not user_id:
                logging.error("Missing user_id in request headers.")
                return {"success": False, "error": "Authentication required."}

            # Validate file
            validation_result = await self.file_validator.validate_file(user_id, file_name, file_size, file_type, file_hash)

            if not validation_result["success"]:
                logging.error(f"File validation failed: {validation_result['error']}")
                return validation_result

            # Publish to Kafka for processing
            upload_request = {
                "user_id": user_id,
                "file_name": file_name,
                "file_size": file_size,
                "file_type": file_type,
                "file_hash": file_hash
            }

            await self.kafka_handler.publish(self.kafka_topic, upload_request)

            logging.info(f"Upload request for '{file_name}' added to Kafka topic '{self.kafka_topic}'")
            return {"success": True, "message": "File upload request received."}

        except Exception as e:
            logging.error(f"Error handling upload request for '{file_name}': {e}")
            return {"success": False, "error": str(e)}