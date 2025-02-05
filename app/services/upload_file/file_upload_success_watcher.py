import logging
import asyncio
from app.core.kafka.kafka_handler import KafkaHandler
from app.core.storage_bin.postgres.postgres_handler import PostgresHandler

class FileUploadSuccessWatcher:
    """
    Watches the Kafka `File_Upload_Success` queue and updates PostgreSQL `files` table.
    """

    def __init__(self, kafka_config: dict, db_url: str):
        """
        Initializes Kafka consumer and PostgreSQL handler.

        Args:
            kafka_config (dict): Kafka connection details.
            db_url (str): PostgreSQL connection string.
        """
        self.kafka = KafkaHandler(**kafka_config)
        self.db = PostgresHandler(db_url)

    async def watch_success_queue(self):
        """
        Continuously listens for successful file uploads and updates PostgreSQL.
        """
        logging.info("Starting File Upload Success Watcher...")

        while True:
            try:
                # Consume messages from the success queue
                success_message = await self.kafka.consume_message("file_upload_success")

                if success_message:
                    logging.info(f"Processing file upload success for {success_message['file_name']}.")

                    await self.update_postgres(success_message)

            except Exception as e:
                logging.error(f"Error in File Upload Success Watcher: {e}")
                await asyncio.sleep(5)  # Avoid rapid failure loops

    async def update_postgres(self, success_message: dict):
        """
        Updates the PostgreSQL `files` table with the uploaded file metadata.

        Args:
            success_message (dict): File metadata from Kafka.
        """
        try:
            user_id = success_message["user_id"]
            file_name = success_message["file_name"]
            relative_path = success_message["relative_path"]
            upload_id = success_message["upload_id"]
            file_size = success_message["file_size"]
            mime_type = success_message["mime_type"]
            etag = success_message["etag"]
            location = success_message["location"]
            bucket = success_message["bucket"]
            object_key = success_message["object_key"]
            version_id = success_message.get("version_id")  # Optional, only if versioning is enabled

            # Store the final MinIO object path
            minio_path = f"{bucket}/{object_key}"

            # Insert into files table with MinIO metadata
            await self.db.insert_file_metadata(
                user_id=user_id,
                file_name=file_name,
                file_path=minio_path,  # Store MinIO path directly
                file_size=file_size,
                file_hash=etag,  # Use MinIO ETag for integrity verification
                mime_type=mime_type,
                version_id=version_id  # Optional: Store version ID if applicable
            )

            await self.db.delete_multipart_upload(upload_id)

            logging.info(f"Successfully updated PostgreSQL for file: {file_name}, Path: {minio_path}")

        except Exception as e:
            logging.error(f"Failed to update PostgreSQL for {success_message['file_name']}: {e}")