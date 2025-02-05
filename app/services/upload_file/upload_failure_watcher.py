import logging
import asyncio
import os
from app.core.kafka.kafka_handler import KafkaHandler
from app.services.upload_file.upload_failure_processor import UploadFailureProcessor

class UploadFailureWatcher:
    """
    Watches Kafka failure queues and routes failed uploads to the processor.
    """

    def __init__(self, kafka_config: dict):
        """
        Initializes Kafka consumer and processor.

        Args:
            kafka_config (dict): Kafka connection details.
        """
        self.kafka = KafkaHandler(**kafka_config)
        self.processor = UploadFailureProcessor(kafka_config)

        # Read backoff delay (in seconds) from .env
        self.delay_seconds = int(os.getenv("UPLOAD_RETRY_DELAY", 10))

    async def watch_failures(self):
        """
        Continuously listens for failed uploads and processes them.
        """
        logging.info("Starting failure watcher for File Upload Failures.")

        while True:
            try:
                # Consume messages from both failure queues
                failure_request = await self.kafka.consume_message("file_upload_failures")
                delayed_request = await self.kafka.consume_message("file_upload_failures_delayed")

                # Handle immediate failures
                if failure_request:
                    logging.info(f"Processing immediate failure for {failure_request['file_name']}.")
                    await self.processor.process_failed_upload(failure_request)

                # Handle delayed failures with backoff
                if delayed_request:
                    logging.info(f"Processing delayed failure for {delayed_request['file_name']} after {self.delay_seconds}s delay.")
                    await asyncio.sleep(self.delay_seconds)  # Backoff before retrying
                    await self.processor.process_failed_upload(delayed_request)

                # Avoid high CPU usage
                await asyncio.sleep(2)

            except Exception as e:
                logging.error(f"Error in Upload Failure Watcher: {e}")
                await asyncio.sleep(5)  # Avoid rapid failure loops