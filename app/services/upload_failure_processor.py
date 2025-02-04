import logging
import os
from app.core.kafka.kafka_handler import KafkaHandler

class UploadFailureProcessor:
    """
    Processes failed file uploads and determines whether to retry or discard.
    """

    def __init__(self, kafka_config: dict):
        """
        Initializes Kafka producer for re-queuing failed uploads.

        Args:
            kafka_config (dict): Kafka connection details.
        """
        self.kafka = KafkaHandler(**kafka_config)
        self.max_retries = int(os.getenv("MAX_UPLOAD_RETRIES", 3))

    async def process_failed_upload(self, upload_request: dict):
        """
        Processes a failed upload request.

        Args:
            upload_request (dict): File upload request with retry metadata.
        """
        try:
            file_name = upload_request["file_name"]
            retries = upload_request.get("retries", 0)

            # Check if retry attempts are within the allowed limit
            if retries < self.max_retries:
                # Increment retry count and send back to upload request queue
                upload_request["retries"] += 1
                await self.kafka.add_to_queue("file_upload_requests", upload_request)
                logging.info(f"Retrying upload for {file_name}. Attempt {upload_request['retries']}/{self.max_retries}.")
            else:
                # Max retries exceeded, move to permanent failure queue
                #TODO: Add logic to handle permanent failure
                logging.error(f"Upload for {file_name} failed after {self.max_retries} retries. Moving to failure queue.")

        except Exception as e:
            logging.error(f"Error processing failed upload: {e}")