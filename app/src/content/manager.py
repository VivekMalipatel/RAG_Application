from typing import List, Dict, Optional
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from .processors import ContentProcessor
from ..utils.logger import setup_logger
from ..utils.config import load_config

logger = setup_logger(__name__)

class ContentManager:
    def __init__(self, vector_store: Optional[Chroma] = None):
        self.config = load_config()
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = vector_store or Chroma(
            embedding_function=self.embeddings,
            persist_directory=self.config["content"]["vector_store_path"]
        )
        self.processor = ContentProcessor(
            chunk_size=self.config["content"]["chunk_size"],
            chunk_overlap=self.config["content"]["chunk_overlap"]
        )
        logger.info("ContentManager initialized")

    async def process_document(self, content: str, metadata: Dict) -> List[str]:
        """Process a document and store it in the vector store"""
        try:
            # Split content into chunks
            chunks = self.processor.split_content(content)
            
            # Create documents with metadata
            documents = self.processor.create_documents(chunks, metadata)
            
            # Add to vector store
            ids = self.vector_store.add_documents(documents)
            logger.info(f"Processed document with {len(chunks)} chunks")
            return ids
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            raise

    def get_next_chunk(self, topic: str, previous_chunks: List[str] = None) -> Dict:
        """Get the next logical chunk of content for teaching"""
        try:
            # Filter out previously covered chunks
            filter_dict = {"topic": topic}
            if previous_chunks:
                filter_dict["chunk_id"] = {"$nin": previous_chunks}
            
            # Get relevant chunks
            results = self.vector_store.similarity_search(
                topic,
                k=1,
                filter=filter_dict
            )
            
            if not results:
                logger.warning(f"No more chunks available for topic: {topic}")
                return None
            
            document = results[0]
            return {
                'content': document.page_content,
                'metadata': document.metadata,
                'chunk_id': document.metadata.get('chunk_id')
            }
        except Exception as e:
            logger.error(f"Error retrieving next chunk: {str(e)}")
            raise

    def get_related_content(self, query: str, n_results: int = 3) -> List[Dict]:
        """Retrieve related content based on a query"""
        try:
            results = self.vector_store.similarity_search(query, k=n_results)
            return [
                {
                    'content': doc.page_content,
                    'metadata': doc.metadata
                }
                for doc in results
            ]
        except Exception as e:
            logger.error(f"Error retrieving related content: {str(e)}")