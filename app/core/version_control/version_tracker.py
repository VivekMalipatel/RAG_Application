from app.core.version_control.metadata_store import MetadataStore

class VersionTracker:
    """Handles document versioning."""

    def __init__(self):
        self.metadata_store = MetadataStore()

    async def track_version(self, user_id: str, doc_name: str, doc_content: bytes):
        """Tracks a document's version in PostgreSQL."""
        return self.metadata_store.add_document(user_id, doc_name, doc_content) 