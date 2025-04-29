Okay, here is a comprehensive plan to implement the in-place document replacement strategy using deterministic IDs derived from `source` and `filename`, integrating it into your existing architecture plan.

**Goal:** When a new file/data arrives with the same `source` and `filename` as an existing entry, automatically replace the old vectors and metadata with the new ones in the FAISS index.

**Phase 1: Core Implementation (QueueConsumer & VectorStore)**

1.  **Deterministic `doc_id` Generation:**
    *   **Location:** queue_consumer.py (within the method handling processed data before calling `vector_store.add_document`).
    *   **Action:**
        *   Retrieve `source` and `filename` from the processed item's metadata. Ensure these fields are mandatory or handled gracefully if missing.
        *   Generate a stable `doc_id` using `hashlib.sha256(f"{source}:{filename}".encode()).hexdigest()`.
        *   Pass this `stable_doc_id` along with embeddings and metadata for further processing.

2.  **Modify `QueueConsumer` Processing Logic:**
    *   **Location:** queue_consumer.py (within the method handling processed data).
    *   **Action:**
        *   **Before** calling `vector_store.add_document`:
            *   Call `removed = self.vector_store.remove_document(stable_doc_id)`.
            *   Log whether an existing document was found and removed (`logger.info(f"Removed existing document {stable_doc_id}..."`) or not (`logger.info(f"No existing document {stable_doc_id} found. Adding as new.")`).
        *   Call `vectors_added = self.vector_store.add_document(doc_id=stable_doc_id, embeddings=..., metadata=...)`.
        *   **After** successful `add_document`:
            *   Call `self.vector_store.save()` to persist the index and mappings.
        *   **Error Handling:** Wrap the `remove_document`, `add_document`, and `save` calls in a `try...except` block. Log errors clearly. If `remove_document` succeeds but `add_document` or `save` fails, log a critical warning about potential data inconsistency (document removed but not replaced).

3.  **Update `VectorStore.add_document`:**
    *   **Location:** vector_store.py
    *   **Action:**
        *   Ensure the method accepts `doc_id` as a parameter.
        *   Use this `doc_id` as the key when storing document info in `self.doc_to_vectors`.
        *   When adding vector IDs to `self.id_to_doc`, store the provided `doc_id` (e.g., `self.id_to_doc[int(vid)] = {'doc_id': doc_id, 'page': ..., 'offset': ...}`).

4.  **Implement `VectorStore.remove_document`:**
    *   **Location:** vector_store.py
    *   **Action:**
        *   Method signature: `def remove_document(self, doc_id: str) -> bool:`
        *   Look up `doc_id` in `self.doc_to_vectors`.
        *   If `doc_id` not found, log debug/info message and return `False`.
        *   If found:
            *   Retrieve the `doc_info = self.doc_to_vectors.pop(doc_id)`.
            *   Collect all associated FAISS vector IDs (`ids_to_remove`) from `doc_info['pages']`.
            *   If `ids_to_remove` is empty, log a warning and return `False` (mapping existed but no vectors).
            *   Call `num_removed = self.index.remove_ids(np.array(ids_to_remove, dtype='int64'))`. Handle potential exceptions from FAISS. Log discrepancies between `len(ids_to_remove)` and `num_removed`.
            *   Iterate through `ids_to_remove` and remove corresponding keys from `self.id_to_doc`.
            *   Log success, including the number of vectors removed.
            *   Return `True`.
        *   Add robust `try...except` around FAISS operations. If removal fails, consider re-inserting `doc_info` back into `self.doc_to_vectors` to maintain mapping consistency, log the error, and re-raise.

5.  **Update `VectorStore.save` and `VectorStore.load`:**
    *   **Location:** vector_store.py
    *   **Action:**
        *   Ensure `self.doc_to_vectors`, `self.id_to_doc`, and `self.next_id` are saved to the mapping file (e.g., `document_mapping.pkl`).
        *   Ensure these mappings are correctly loaded and restored when `load()` is called.
        *   Add checks during `load` (e.g., verify loaded embedding dimension matches configured dimension) to prevent loading inconsistent data.

**Phase 2: Refinements and Production Considerations**

6.  **Metadata Integrity:**
    *   **Action:** Add validation (e.g., using Pydantic models) at the API ingestion point (`app/api/routes/ingest.py` - assuming this exists based on your OpenAPI spec) and potentially within the `QueueConsumer` to ensure `source` and `filename` are always present in the metadata when expected.

7.  **Concurrency Control:**
    *   **Assessment:** Determine if multiple `QueueConsumer` instances will run concurrently and access the same `VectorStore` instance/files.
    *   **Action (If Concurrent):** Implement locking. A simple approach is using `filelock` around the `remove -> add -> save` sequence in the `QueueConsumer`.
        ```python
        # In QueueConsumer, around the vector store operations
        from filelock import FileLock
        lock_path = os.path.join(self.vector_store.index_dir, "vectorstore.lock")
        with FileLock(lock_path):
            removed = self.vector_store.remove_document(stable_doc_id)
            # ... log removal ...
            vectors_added = self.vector_store.add_document(...)
            # ... log addition ...
            self.vector_store.save()
        ```

8.  **Index Maintenance (Long-Term):**
    *   **Plan:** Recognize that FAISS HNSW `remove_ids` marks vectors for deletion but doesn't immediately reclaim space or restructure the graph optimally.
    *   **Action:** Schedule periodic maintenance (e.g., weekly, monthly, or triggered by fragmentation metrics) to rebuild the index. This involves:
        *   Loading the current index and mappings.
        *   Creating a new, empty FAISS index (`IndexIDMap(IndexHNSWFlat(...))`).
        *   Iterating through `self.doc_to_vectors` and `self.id_to_doc` to re-add all *active* vectors (those not marked for deletion) to the new index using their original IDs.
        *   Saving the new index and the potentially cleaned mappings.

9.  **Monitoring and Logging:**
    *   **Action:** Ensure detailed logging for:
        *   `doc_id` generation.
        *   Document removal attempts (found/not found).
        *   Document addition.
        *   Save operations.
        *   Any errors encountered during the process.
        *   Discrepancies reported by FAISS (e.g., `remove_ids` count mismatch).

**Phase 3: Testing**

10. **Unit Tests:**
    *   Test `VectorStore.add_document`, `VectorStore.remove_document`, `VectorStore.save`, `VectorStore.load` in isolation.
    *   Test `doc_id` generation logic.
11. **Integration Tests:**
    *   Test the `QueueConsumer`'s ability to correctly call `remove`, `add`, and `save` in sequence.
    *   Simulate receiving a new document.
    *   Simulate receiving the same document again (triggering replacement).
    *   Verify search results only reflect the latest version after replacement.
    *   Test error scenarios (e.g., failure during `add` after successful `remove`).
    *   Test concurrent processing if applicable.

This plan integrates the deterministic replacement strategy directly into your planned `QueueConsumer` and `VectorStore` components, leveraging the `source` and `filename` metadata for identification.