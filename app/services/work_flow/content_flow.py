from typing import List, Dict, Optional
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

class ContentManager:
    def __init__(self, vector_store: Optional[Chroma] = None):
        """
        Initialize ContentManager with an optional existing vector store.
        If none provided, it will create a new one.
        """
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = vector_store or Chroma(
            embedding_function=self.embeddings,
            persist_directory="./content_store"
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

    def process_document(self, content: str, metadata: Dict) -> List[str]:
        """
        Process a document and store it in the vector store.
        Returns list of chunk IDs for retrieval.
        """
        # Split content into teachable chunks
        chunks = self.text_splitter.split_text(content)
        
        # Create documents with metadata
        documents = [
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
        
        # Add to vector store
        ids = self.vector_store.add_documents(documents)
        return ids

    def get_next_chunk(self, topic: str, previous_chunks: List[str] = None) -> Dict:
        """
        Get the next logical chunk of content for teaching.
        Takes into account previously covered chunks.
        """
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
            return None
            
        document = results[0]
        return {
            'content': document.page_content,
            'metadata': document.metadata,
            'chunk_id': document.metadata.get('chunk_id')
        }

    def get_related_content(self, query: str, n_results: int = 3) -> List[Dict]:
        """
        Retrieve related content based on a query (e.g., student question)
        """
        results = self.vector_store.similarity_search(query, k=n_results)
        return [
            {
                'content': doc.page_content,
                'metadata': doc.metadata
            }
            for doc in results
        ]

class EnhancedTeachingSession(TeachingSession):
    def __init__(self, student_id: str, subject: str, llm, content_manager: ContentManager):
        super().__init__(student_id, subject, llm)
        self.content_manager = content_manager
        self.covered_chunks = []
        
    def start_topic(self, topic: str) -> str:
        """Start teaching a new topic using processed content"""
        self.current_topic = topic
        self.topic_start_time = datetime.now()
        
        # Get first chunk of content
        chunk_data = self.content_manager.get_next_chunk(topic)
        if not chunk_data:
            return f"I'm sorry, I couldn't find any content for the topic: {topic}"
            
        self.covered_chunks.append(chunk_data['chunk_id'])
        return self._generate_introduction(topic, chunk_data['content'])

    def handle_student_input(self, student_input: str) -> str:
        """Enhanced handling of student input with content retrieval"""
        self.engagement_metrics["responses_given"] += 1
        
        # Check if it's a question
        if "?" in student_input:
            self.engagement_metrics["questions_asked"] += 1
            # Get related content to help answer the question
            related_content = self.content_manager.get_related_content(student_input)
            # Add relevant content to the context
            context = "\n".join([rc['content'] for rc in related_content])
            response = self.teaching_chain.run(
                topic=self.current_topic,
                current_input=f"Context: {context}\nStudent Question: {student_input}",
                chat_history=self.memory.chat_memory.messages
            )
        else:
            response = super().handle_student_input(student_input)
        
        # Check if we should move to next chunk
        if self.should_advance_content():
            next_chunk = self.content_manager.get_next_chunk(
                self.current_topic, 
                self.covered_chunks
            )
            if next_chunk:
                self.covered_chunks.append(next_chunk['chunk_id'])
                response += f"\n\nLet's move on to the next part:\n{self._transition_to_new_content(next_chunk['content'])}"
        
        return response

    def should_advance_content(self) -> bool:
        """
        Determine if we should move to the next chunk of content
        based on engagement metrics and comprehension
        """
        # This is a simple implementation - you can make it more sophisticated
        last_responses = len(self.memory.chat_memory.messages) - len(self.covered_chunks)
        return last_responses >= 3  # Move on after 3 exchanges about current chunk

    def _transition_to_new_content(self, content: str) -> str:
        """Generate a smooth transition to new content"""
        transition_prompt = f"""
        Based on our previous discussion, create a smooth transition to this new content:
        {content}
        Make sure to connect it with what we've discussed before.
        """
        return self.teaching_chain.run(
            topic=self.current_topic,
            current_input=transition_prompt,
            chat_history=self.memory.chat_memory.messages
        )

# Example usage:
def setup_teaching_environment(student_id: str, subject: str, llm):
    content_manager = ContentManager()
    return EnhancedTeachingSession(student_id, subject, llm, content_manager)