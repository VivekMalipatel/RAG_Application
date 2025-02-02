import hashlib
import json

class MetadataManager:
    """Manages relationships between chunks and documents."""

    def __init__(self):
        self.metadata_store = {}

    def generate_chunk_id(self, text: str):
        """Generates a unique hash for each chunk."""
        return hashlib.sha256(text.encode()).hexdigest()

    def store_metadata(self, doc_id, chunk_texts):
        """Stores chunk relationships with the document."""
        chunk_metadata = []
        for chunk in chunk_texts:
            chunk_id = self.generate_chunk_id(chunk)
            chunk_metadata.append({
                "chunk_id": chunk_id,
                "document_id": doc_id,
                "text": chunk
            })
        self.metadata_store[doc_id] = chunk_metadata

    def get_metadata(self, doc_id):
        """Retrieves metadata for a document."""
        return self.metadata_store.get(doc_id, [])

    def save_metadata_to_file(self, filename="metadata.json"):
        """Saves metadata as a JSON file."""
        with open(filename, "w") as f:
            json.dump(self.metadata_store, f, indent=4)