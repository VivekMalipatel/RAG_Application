import asyncio
import pytest
from app.services.file_processor.entity_relation_extractor import EntityRelationExtractor
from app.core.embedding.embedding_handler import EmbeddingHandler
from app.core.graph_db.neo4j.neo4j_search import Neo4jSearchHandler
from app.core.vector_store.qdrant.qdrant_handler import QdrantHandler
from app.core.models.model_handler import ModelRouter
from app.core.models.model_provider import Provider
from app.core.models.model_type import ModelType
from app.config import settings

# Test user_id (Ensure the user_id exists in Neo4j & Qdrant)
TEST_USER_ID = 1234324

@pytest.mark.asyncio
async def test_end_to_end_query():
    """
    End-to-end test for real-time query processing.
    This will:
    - Extract entities/relations
    - Compute embeddings
    - Perform Neo4j search (entities, relationships, knowledge paths)
    - Apply ColBERT reranking
    - Run refined Qdrant hybrid search
    - Generate the final response via LLM
    """

    print("\n===== Starting End-to-End Query Test =====")

    query_text = "Who is Vivek Malipatel?"
    print(f"\n[STEP 1] User Query: {query_text}")

    # Step 1: Extract Entities & Relations
    print("\n[STEP 2] Extracting Entities & Relationships...")
    entity_extractor = EntityRelationExtractor()
    
    extracted_data = await entity_extractor.extract_entities_and_relationships(
        [{"content": query_text, "chunk_metadata": {"user_id": TEST_USER_ID, "document_id": "test_doc", "chunk_number":0, "doc_summary": "This is a test document to search for Vivek Malipatel in knowledge graph", "context": "This is a test document to search for Vivek Malipatel in knowledge graph"}}]
    )  
    print(f"Extracted Data: {extracted_data}")
    
    entities = extracted_data[0]["chunk_metadata"].get("entities", [])
    relationships = extracted_data[0]["chunk_metadata"].get("relationships", [])

    assert entities, "Entity extraction failed."
    assert relationships, "Relationship extraction failed."

    print(f"Extracted Entities: {entities}")
    print(f"Extracted Relationships: {relationships}")

    # Step 3: Compute Embeddings
    print("\n[STEP 3] Generating Embeddings for Query...")
    embedding_handler = EmbeddingHandler(
        provider=Provider.HUGGINGFACE,
        model_name=settings.TEXT_EMBEDDING_MODEL_NAME,
        model_type=ModelType.TEXT_EMBEDDING
    )
    
    entity_embeddings = await embedding_handler.encode_dense([e["text"] for e in entities])
    relation_embeddings = await embedding_handler.encode_dense([r["relation_type"] for r in relationships])

    assert entity_embeddings, "Entity embedding generation failed."
    assert relation_embeddings, "Relationship embedding generation failed."
    
    print("✅ Embeddings generated successfully.")

    # Step 4: Perform Neo4j Searches
    print("\n[STEP 4] Searching Neo4j Knowledge Graph...")
    neo4j_search = Neo4jSearchHandler()
    
    search_results = []
    for entity, embedding in zip(entities, entity_embeddings):
        entity_results = await neo4j_search.search_entities(
            user_id=TEST_USER_ID,
            query_embedding=embedding[:256],
            entity_type=entity["entity_type"],
            limit=5
        )
        search_results.extend(entity_results)

    for relation, embedding in zip(relationships, relation_embeddings):
        relation_results = await neo4j_search.search_relationships(
            user_id=TEST_USER_ID,
            query_embedding=embedding[:256],
            relation_type=relation["relation_type"],
            limit=5
        )
        search_results.extend(relation_results)

    print(f"✅ Retrieved {len(search_results)} results from Neo4j.")

    # Step 5: Perform ColBERT Re-Ranking (Mocked for now)
    print("\n[STEP 5] Applying ColBERT Re-Ranking...")
    reranked_results = sorted(search_results, key=lambda x: x["score"], reverse=True)[:10]  # Mock ranking

    print("✅ ColBERT Re-ranking done.")

    # Step 6: Hybrid Search in Qdrant
    print("\n[STEP 6] Refining Search in Qdrant...")
    qdrant_handler = QdrantHandler()

    refined_query = " ".join([res["text"] for res in reranked_results])
    print(f"Refined Query: {refined_query}")
    dense_embedding = await embedding_handler.encode_dense(refined_query)
    sparse_embedding = await embedding_handler.encode_sparse(refined_query)

    qdrant_results = await qdrant_handler.hybrid_search(
        user_id=TEST_USER_ID,
        query_text=refined_query,
        dense_vector=dense_embedding,
        sparse_vector=sparse_embedding,
        top_k=5
    )

    print(f"✅ Retrieved {len(qdrant_results)} results from Qdrant.")

    # Step 7: Generate Final Answer using LLM
    print("\n[STEP 7] Generating Answer with LLM...")
    llm = ModelRouter(
        provider=Provider.OPENAI,
        model_name="gpt-4o-mini",
        model_type=ModelType.TEXT_GENERATION,
        system_prompt="Answer the question concisely using the retrieved context."
    )

    retrieved_texts = [res.payload["content"] for res in qdrant_results]
    context_text = "\n".join(retrieved_texts)

    llm_prompt = f"""
    User Query: {query_text}
    
    Retrieved Information:
    {context_text}

    Final Answer:
    """
    llm_answer = await llm.generate_text(llm_prompt)

    assert llm_answer, "LLM failed to generate an answer."

    print(f"✅ LLM Response: {llm_answer.strip()}")

    print("\n===== End-to-End Query Test Completed Successfully =====")

    # Final assertion
    assert llm_answer.strip(), "Final answer is empty!"

if __name__ == "__main__":
    asyncio.run(test_end_to_end_query())