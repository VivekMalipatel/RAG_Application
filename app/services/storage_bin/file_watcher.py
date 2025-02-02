import time
import asyncio
import logging
import aiofiles
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from app.core.storage_bin.file_handler import FileHandler

class FileEventHandler(FileSystemEventHandler):
    """Handles file system events asynchronously."""

    def __init__(self, user_id):
        self.file_handler = FileHandler()
        self.user_id = user_id

    async def on_created(self, event):
        """Triggered when a new file is created."""
        if event.is_directory:
            return
        logging.info(f"New file detected: {event.src_path}")

        async with aiofiles.open(event.src_path, "rb") as f:
            file_data = await f.read()
            await self.file_handler.save_file(self.user_id, file_data, event.src_path)

    async def on_deleted(self, event):
        """Triggered when a file is deleted."""
        if event.is_directory:
            return
        logging.info(f"File deleted: {event.src_path}")

class DirectoryWatcher:
    """Monitors a directory asynchronously for file changes."""

    def __init__(self, directory_to_watch, user_id):
        self.directory_to_watch = directory_to_watch
        self.event_handler = FileEventHandler(user_id)
        self.observer = Observer()

    async def start(self):
        """Starts the directory watcher asynchronously."""
        self.observer.schedule(self.event_handler, self.directory_to_watch, recursive=True)
        self.observer.start()
        logging.info(f"Started watching directory: {self.directory_to_watch}")

        try:
            while True:
                await asyncio.sleep(5)
        except KeyboardInterrupt:
            self.observer.stop()
            logging.info("Stopping directory watcher.")

        self.observer.join()