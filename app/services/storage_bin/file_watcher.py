import os
import asyncio
import logging
import signal
from concurrent.futures import ThreadPoolExecutor
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from app.core.storage_bin.file_handler import FileHandler

class FileEventHandler(FileSystemEventHandler):
    """Handles file system events asynchronously using background tasks."""

    def __init__(self, user_id, loop):
        self.file_handler = FileHandler()
        self.user_id = user_id
        self.loop = loop  # Attach event loop for async execution

    def on_created(self, event):
        """Triggered when a new file is created."""
        if event.is_directory:
            return
        logging.info(f"New file detected: {event.src_path}")
        self.loop.create_task(self.process_file(event.src_path))  # Offload to async task

    async def process_file(self, file_path):
        """Reads & uploads file asynchronously to MinIO."""
        try:
            file_name = os.path.basename(file_path)

            # Read file in a separate thread (fix for binary data reading)
            file_data = await asyncio.to_thread(self.read_file, file_path)

            if file_data:
                minio_path = await self.file_handler.save_file(self.user_id, file_data, file_name)
                if minio_path:
                    logging.info(f"File '{file_name}' uploaded successfully: {minio_path}")
                else:
                    logging.error(f"Failed to upload file '{file_name}'")
            else:
                logging.error(f"Empty file '{file_name}', skipping upload.")

        except Exception as e:
            logging.error(f"Error processing file '{file_path}': {e}")

    @staticmethod
    def read_file(file_path):
        """Reads binary file content (used for async threading)."""
        with open(file_path, "rb") as f:
            return f.read()

    def on_deleted(self, event):
        """Triggered when a file is deleted."""
        if event.is_directory:
            return
        logging.info(f"File deleted: {event.src_path}")

class DirectoryWatcher:
    """Monitors a directory asynchronously for file changes using watchdog."""

    def __init__(self, directory_to_watch, user_id):
        self.directory_to_watch = directory_to_watch
        self.user_id = user_id
        self.loop = asyncio.new_event_loop()  # Create a dedicated event loop
        self.event_handler = FileEventHandler(user_id, self.loop)
        self.observer = Observer()
        self.executor = ThreadPoolExecutor(max_workers=2)  # Limit threads for safety
        self.stop_event = asyncio.Event()  # Stop event for graceful shutdown

    async def start(self):
        """Starts the directory watcher asynchronously."""
        if not os.path.exists(self.directory_to_watch):
            logging.error(f"Directory '{self.directory_to_watch}' does not exist. Exiting watcher.")
            return

        try:
            self.observer.schedule(self.event_handler, self.directory_to_watch, recursive=True)
            self.executor.submit(self.observer.start)  # Run in a separate thread
            logging.info(f"Watching directory: {self.directory_to_watch}")

            # Handle graceful shutdown on SIGINT or SIGTERM (Only on Unix)
            if hasattr(signal, "SIGTERM"):
                for sig in (signal.SIGINT, signal.SIGTERM):
                    self.loop.add_signal_handler(sig, self.stop_event.set)

            await self.stop_event.wait()  # Wait for shutdown event

        except Exception as e:
            logging.error(f"Error starting observer: {e}")

        finally:
            self.shutdown()

    def shutdown(self):
        """Stops the observer and cleans up resources."""
        logging.info("Stopping directory watcher...")
        self.observer.stop()
        self.executor.shutdown(wait=True)
        self.observer.join()
        logging.info("Directory watcher stopped gracefully.")