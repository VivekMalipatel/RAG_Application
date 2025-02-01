from app.core.vectorStore.qdrant import QdrantVectorStore

# Initialize QdrantVectorStore
qdrant_url = "http://localhost:6333"  # Ensure this is correct and the server is running
base_collection_name = "test_collection"
vector_store = QdrantVectorStore(qdrant_url, base_collection_name)

# Define test parameters
user_id = "user123"
model_dense = "BAAI/bge-base-en"
model_late_interaction = "colbert-ir/colbertv2.0"
doc_id = 1
text = "This is a test document."
query = "test query"
top_k = 5

# Store a document
vector_store.store_document(user_id, model_dense, model_late_interaction, doc_id, text)

# Perform a hybrid search
results = vector_store.hybrid_search(user_id, model_dense, model_late_interaction, query, top_k)

# Print results
print("Search Results:", results)
