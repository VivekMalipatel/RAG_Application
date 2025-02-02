from app.core.vectorstore.index_manager import IndexManager
import asyncio

class BatchIndexer:
    """Handles batch document indexing for efficiency."""

    def __init__(self, collection_name="test_collection"):
        self.index_manager = IndexManager()
        self.collection_name = collection_name

    async def index_batch(self, documents):
        """Indexes multiple documents in batches."""
        text_chunks = [doc["text"] for doc in documents]

        # Split documents into batches of 10
        batch_size = 10
        for i in range(0, len(text_chunks), batch_size):
            batch = text_chunks[i:i + batch_size]
            print(f"🔹 Indexing batch {i//batch_size + 1}")
            await asyncio.to_thread(self.index_manager.index_document, self.collection_name, batch)