import os
import numpy as np
import faiss
import pickle
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, index_dir: str = "storage/faiss_indexes", embedding_dim: int = 128):
        self.index_dir = index_dir
        self.embedding_dim = embedding_dim
        self.index = None
        self.doc_to_vectors = {}
        self.page_mapping = {}
        self.next_id = 0
        self.id_to_doc = {}
        os.makedirs(index_dir, exist_ok=True)
        logger.info(f"VectorStore initialized with index directory: {index_dir}")
        
    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms
        
    def initialize_index(self, use_gpu: bool = False):
        if self.index is None:
            hnsw = faiss.IndexHNSWFlat(self.embedding_dim, 32, faiss.METRIC_INNER_PRODUCT)
            hnsw.hnsw.efConstruction = 128
            hnsw.hnsw.efSearch = 128
            idmap = faiss.IndexIDMap(hnsw)
            self.index = idmap
            if use_gpu and faiss.get_num_gpus() > 0:
                co = faiss.GpuClonerOptions()
                co.shard = True
                self.index = faiss.index_cpu_to_all_gpus(self.index, co)
                logger.info("Using multi-GPU for FAISS")
            logger.info(f"Initialized FAISS IDMap(HNSW) with dimension {self.embedding_dim} for cosine similarity")
        return self.index
    
    def add_document(self, doc_id: str, embeddings: List[List[List[float]]], metadata: Dict[str, Any] = None):
        if self.index is None:
            self.initialize_index()
        
        total_vectors_added = 0
        page_indices = {}
        
        for page_idx, page_embeddings in enumerate(embeddings):
            if not page_embeddings:
                continue
                
            page_embeddings_np = np.array(page_embeddings, dtype='float32')
            page_embeddings_np = self._normalize_vectors(page_embeddings_np)
            
            n = page_embeddings_np.shape[0]
            ids = np.arange(self.next_id, self.next_id + n, dtype='int64')
            self.index.add_with_ids(page_embeddings_np, ids)
            for vid in ids:
                self.id_to_doc[int(vid)] = (doc_id, page_idx)
            start_id = int(ids[0])
            end_id = int(ids[-1]) + 1
            vectors_added = n
            self.next_id += n
            
            page_indices[str(page_idx)] = {
                'start_id': start_id,
                'end_id': end_id,
                'count': vectors_added
            }
            
            total_vectors_added += vectors_added
        
        self.doc_to_vectors[doc_id] = {
            'count': int(total_vectors_added),
            'metadata': metadata or {},
            'pages': page_indices,
            'raw_embeddings': embeddings
        }
        
        logger.info(f"Added document {doc_id} with {total_vectors_added} vectors across {len(page_indices)} pages")
        return total_vectors_added
    
    def search(self, query_vectors: List[List[List[float]]], k: int = 10) -> List[Dict]:
        if self.index is None:
            logger.error("Index not initialized")
            return []
            
        try:
            try:
                self.index.index.hnsw.efSearch = 256
            except AttributeError:
                logger.debug("efSearch tuning skipped; index might not be HNSW type")
                pass

            arr = np.array(query_vectors, dtype='float32')
            if arr.ndim != 3:
                raise ValueError(f"Expected 3D input with shape [pages][chunks][dim], got {arr.ndim}D shape {arr.shape}")

            if arr.shape[2] != self.embedding_dim:
                raise ValueError(f"Expected embedding dimension {self.embedding_dim}, got {arr.shape[2]}")

            pages, chunks, dim = arr.shape
            arr = arr.reshape(pages * chunks, dim)
                
            arr = self._normalize_vectors(arr)
            
            total = self.index.ntotal if hasattr(self.index, 'ntotal') else self.next_id
            expanded_k = min(k * 10, total) if total > 0 else k

            distances, indices = self.index.search(arr, expanded_k)

            doc_scores = {}
            
            for q_idx in range(arr.shape[0]):
                for i in range(min(expanded_k, len(indices[q_idx]))):
                    vid = int(indices[q_idx][i])
                    if vid < 0:
                        continue
                    similarity = float(distances[q_idx][i])
                    info = self.id_to_doc.get(vid)
                    if not info:
                        continue
                        
                    doc_id, page_num = info
                    # If we've removed that document, skip it
                    if doc_id not in self.doc_to_vectors:
                        continue
                        
                    if doc_id not in doc_scores or similarity > doc_scores[doc_id]['score']:
                        meta = self.doc_to_vectors.get(doc_id, {}).get('metadata', {}).copy()
                        result = {
                            'doc_id': doc_id, 
                            'score': similarity, 
                            'metadata': meta
                        }
                        if page_num is not None:
                            result['page'] = page_num
                        doc_scores[doc_id] = result
            
            results = sorted(doc_scores.values(), key=lambda x: x['score'], reverse=True)
            
            return results[:k]
        except Exception as e:
            logger.error(f"Error during vector search: {str(e)}", exc_info=True)
            return []
    
    def save(self):
        if self.index is None:
            logger.warning("No index to save")
            return False
            
        try:
            os.makedirs(self.index_dir, exist_ok=True)
            index_path = os.path.join(self.index_dir, "faiss.index")
            mapping_path = os.path.join(self.index_dir, "document_mapping.pkl")
            
            idx_to_save = faiss.index_gpu_to_cpu(self.index) if hasattr(self.index, 'getDevice') else self.index
            faiss.write_index(idx_to_save, index_path)
            
            with open(mapping_path, 'wb') as f:
                pickle.dump(self.doc_to_vectors, f)
            id_map_path = os.path.join(self.index_dir, "id_mapping.pkl")
            with open(id_map_path, 'wb') as f:
                pickle.dump(self.id_to_doc, f)
            
            logger.info(f"Saved index with {self.index.ntotal} vectors and {len(self.doc_to_vectors)} documents")
            return True
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}", exc_info=True)
            return False
    
    def load(self):
        index_path = os.path.join(self.index_dir, "faiss.index")
        mapping_path = os.path.join(self.index_dir, "document_mapping.pkl")
        
        if not os.path.exists(index_path) or not os.path.exists(mapping_path):
            logger.info("No saved index on disk; starting with an empty FAISS index")
            self.initialize_index()
            return True
            
        try:
            self.index = faiss.read_index(index_path, faiss.IO_FLAG_MMAP)
            
            with open(mapping_path, 'rb') as f:
                self.doc_to_vectors = pickle.load(f)
            id_map_path = os.path.join(self.index_dir, "id_mapping.pkl")
            if os.path.exists(id_map_path):
                with open(id_map_path, 'rb') as f:
                    self.id_to_doc = pickle.load(f)
                self.next_id = max(self.id_to_doc.keys(), default=-1) + 1
            else:
                self.id_to_doc = {}
                self.next_id = 0
                for doc_id, info in self.doc_to_vectors.items():
                    for page_idx_str, page_info in info.get('pages', {}).items():
                        start_id = page_info['start_id']
                        end_id = page_info['end_id']
                        for vid in range(start_id, end_id):
                            self.id_to_doc[vid] = (doc_id, int(page_idx_str))
                            if vid >= self.next_id:
                                self.next_id = vid + 1
             
            logger.info(f"Loaded index with {self.index.ntotal} vectors and {len(self.doc_to_vectors)} documents")
            return True
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}", exc_info=True)
            return False
            
    def get_stats(self):
        if self.index is None:
            return {"status": "not_initialized"}
        
        total_pages = 0
        for doc_id, doc_info in self.doc_to_vectors.items():
            total_pages += len(doc_info.get('pages', {}))
            
        return {
            "total_vectors": self.index.ntotal,
            "total_documents": len(self.doc_to_vectors),
            "total_pages": total_pages,
            "embedding_dimension": self.embedding_dim,
            "index_type": type(self.index).__name__
        }

    def remove_document(self, doc_id: str) -> bool:
        doc_info = self.doc_to_vectors.pop(doc_id, None)
        if not doc_info:
            logger.warning(f"Document {doc_id} not found")
            return False
        ids_to_remove = []
        for page_info in doc_info['pages'].values():
            ids_to_remove.extend(range(page_info['start_id'], page_info['end_id']))
        
        for vid in ids_to_remove:
            self.id_to_doc.pop(vid, None)
            
        logger.info(f"Lazy-removed document {doc_id} with {len(ids_to_remove)} vectors")
        return True

    def search_batch(self, queries: List[List[List[float]]], k: int = 10) -> List[List[Dict]]:
        if self.index is None:
            logger.error("Index not initialized")
            return []
            
        try:
            arr = np.array(queries, dtype='float32')
            if arr.ndim != 3:
                raise ValueError(f"Expected 3D input with shape [pages][chunks][dim], got {arr.ndim}D shape {arr.shape}")

            if arr.shape[2] != self.embedding_dim:
                raise ValueError(f"Expected embedding dimension {self.embedding_dim}, got {arr.shape[2]}")

            batch_results = []
            for page_idx in range(arr.shape[0]):
                page_embeddings = arr[page_idx]

                page_embeddings = self._normalize_vectors(page_embeddings)

                total = self.index.ntotal if hasattr(self.index, 'ntotal') else self.next_id
                expanded_k = min(k * 10, total)
                distances, indices = self.index.search(page_embeddings, expanded_k)

                doc_scores = {}
                for q_i in range(page_embeddings.shape[0]):
                    for i, vid in enumerate(indices[q_i]):
                        if vid < 0:
                            continue
                        similarity = float(distances[q_i][i])
                        info = self.id_to_doc.get(int(vid))
                        if not info:
                            continue
                        doc_id, page_num = info
                        if doc_id not in self.doc_to_vectors:
                            continue
                            
                        if doc_id not in doc_scores or similarity > doc_scores[doc_id]['score']:
                            meta = self.doc_to_vectors.get(doc_id, {}).get('metadata', {}).copy()
                            res = {'doc_id': doc_id, 'score': similarity, 'metadata': meta}
                            if page_num is not None:
                                res['page'] = page_num
                            doc_scores[doc_id] = res
                
                results = sorted(doc_scores.values(), key=lambda x: x['score'], reverse=True)[:k]
                batch_results.append(results)
            
            return batch_results
        except Exception as e:
            logger.error(f"Error during batch vector search: {str(e)}", exc_info=True)
            raise

    def rebuild_index(self, use_gpu: bool = False):
        if not self.doc_to_vectors:
            logger.warning("No documents in store to rebuild index")
            return False
            
        try:
            logger.info("Starting index rebuild...")
            
            current_docs = list(self.doc_to_vectors.items())
            
            old_index = self.index
            self.index = None
            self.initialize_index(use_gpu=use_gpu)
            
            self.id_to_doc = {}
            self.next_id = 0
            
            saved_docs = self.doc_to_vectors.copy()
            self.doc_to_vectors = {}
            
            skipped_docs = []
            successful_docs = 0
            
            for doc_id, doc_info in current_docs:
                if 'raw_embeddings' not in doc_info:
                    skipped_docs.append(doc_id)
                    logger.warning(f"Cannot rebuild document {doc_id} - missing raw embeddings")
                    continue
                
                embeddings = doc_info['raw_embeddings']
                metadata = doc_info['metadata']
                
                self.add_document(doc_id, embeddings, metadata)
                successful_docs += 1

            for doc_id in skipped_docs:
                self.doc_to_vectors[doc_id] = saved_docs[doc_id]
            
            logger.info(f"Index rebuilt successfully with {self.index.ntotal} vectors - "
                      f"{successful_docs} documents added, {len(skipped_docs)} documents skipped")
            return True
            
        except Exception as e:
            logger.error(f"Error rebuilding index: {str(e)}", exc_info=True)
            return False
            
    def should_rebuild(self, threshold: float = 0.2) -> bool:
        if self.index is None or self.index.ntotal == 0:
            return False

        active_vectors = 0
        for doc_info in self.doc_to_vectors.values():
            active_vectors += doc_info.get('count', 0)
            
        total_vectors = self.index.ntotal
        deleted_vectors = total_vectors - active_vectors
        deleted_ratio = deleted_vectors / total_vectors if total_vectors > 0 else 0
        
        if deleted_ratio >= threshold:
            logger.info(f"Index rebuild recommended: {deleted_vectors}/{total_vectors} vectors deleted ({deleted_ratio:.2%})")
            return True
        return False

#TODO: HNSW Removal Limitations
# python
# def remove_document(self, doc_id: str) -> bool:
#     ...
#     self.index.remove_ids(np.array(ids_to_remove, dtype='int64'))
# While functional, HNSW+IDMap removal leaves "holes" in graph structure
# For high-churn systems, consider periodic index rebuilds

#TODO :GPU-Persistence Edge Case
# python
# idx_to_save = faiss.index_gpu_to_cpu(self.index) if hasattr(self.index, 'getDevice') else self.index
# Needs validation for multi-GPU sharded indices
# Test with faiss.StandardGpuResources.setTempMemory(...) for large datasets

#TODO: ID Mapping Corruption
# Scenario: Partial index load after failed save
# Detection:
# python
# if os.path.exists(id_map_path):
#     # Load proper mapping
# else:
#     # Rebuild from document metadata

#TODO:GPU-CPU Compatibility
# Solution:
# python
# idx_to_save = faiss.index_gpu_to_cpu(self.index)  # Before serialization

#TODO:Sparse ID Handling
# Approach:
# python
# self.next_id = max(self.id_to_doc.keys(), default=-1) + 1

#TODO:Optimization Guidelines
# Batch Size Tuning:
# Add Documents: 1,000-10,000 vectors/batch
# Search: 100-500 queries/batch
# Memory Configuration:
# python
# # For GPU indices
# faiss.StandardGpuResources.setTempMemory(1024*1024*1024)  # 1GB
# Recall/Speed Tradeoff:
# python
# self.index.index.hnsw.efSearch = 256  # 64-512 range
# This implementation provides production-grade vector search capabilities for multi-modal embeddings while maintaining flexibility for both small-scale and large-scale deployments.

#TODO:Failure Recovery
# 1. Crash Consistency
# Write-Ahead Logging: Optional addition for atomic operations

# Checksum Verification: MD5 hashes of index segments

# Partial Load Handling: Graceful degradation for corrupt files

# 2. Distributed Deployment
# Sharding Strategy:

# python
# co = faiss.GpuClonerOptions()
# co.shard = True
# Consistent Hashing: Automatic query routing to shards

# Replication: 3x replication factor for fault tolerance