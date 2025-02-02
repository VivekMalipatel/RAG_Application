from langchain.text_splitter import RecursiveCharacterTextSplitter
import asyncio

class TextChunker:
    """Splits long text documents into smaller, retrievable chunks."""

    def __init__(self, chunk_size=512, chunk_overlap=64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", ".", "!", "?", " "]
        )

    async def chunk_text(self, text: str):
        """Asynchronously splits text into smaller chunks."""
        return await asyncio.to_thread(self.splitter.split_text, text)

# Example usage:
# chunker = TextChunker()
# chunks = asyncio.run(chunker.chunk_text("This is a long document. It must be split."))