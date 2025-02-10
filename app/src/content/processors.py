from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

class ContentProcessor:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def split_content(self, content: str) -> List[str]:
        """Split content into teachable chunks"""
        return self.text_splitter.split_text(content)
    
    def create_documents(self, chunks: List[str], metadata: Dict) -> List[Document]:
        """Create document objects from chunks"""
        return [
            Document(
                page_content=chunk,
                metadata={
                    **metadata,
                    'chunk_index': i,
                    'total_chunks': len(chunks)
                }
            )
            for i, chunk in enumerate(chunks)
        ]