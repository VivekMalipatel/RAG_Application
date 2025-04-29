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
        
    def initialize_index(self, use_gpu: bool = False):
        if self.index is None:
            hnsw = faiss.IndexHNSWFlat(self.embedding_dim, 32)
            hnsw.hnsw.efConstruction = 128
            hnsw.hnsw.efSearch = 128
            idmap = faiss.IndexIDMap(hnsw)
            self.index = idmap
            if use_gpu and faiss.get_num_gpus() > 0:
                co = faiss.GpuClonerOptions()
                co.shard = True
                self.index = faiss.index_cpu_to_all_gpus(self.index, co)
                logger.info("Using multi-GPU for FAISS")
            logger.info(f"Initialized FAISS IDMap(HNSW) with dimension {self.embedding_dim}")
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
            'pages': page_indices
        }
        
        logger.info(f"Added document {doc_id} with {total_vectors_added} vectors across {len(page_indices)} pages")
        return total_vectors_added
    
    def search(self, query_vectors: List[List[float]], k: int = 10) -> List[Dict]:
        if self.index is None:
            logger.error("Index not initialized")
            return []
            
        try:
            self.index.index.hnsw.efSearch = 128
        except Exception:
            pass
            
        query_np = np.array(query_vectors).astype('float32')
        
        total = self.index.ntotal if hasattr(self.index, 'ntotal') else self.next_id
        expanded_k = min(k * 5, total) if total > 0 else k
        distances, indices = self.index.search(query_np, expanded_k)
        
        doc_scores = {}
        
        for q_idx in range(len(query_vectors)):
            for i in range(expanded_k):
                vid = int(indices[q_idx][i])
                if vid < 0:
                    continue
                distance = float(distances[q_idx][i])
                info = self.id_to_doc.get(vid)
                if not info:
                    continue
                doc_id, page_num = info
                similarity = -distance
                if doc_id not in doc_scores or similarity > doc_scores[doc_id]['score']:
                    meta = self.doc_to_vectors.get(doc_id, {}).get('metadata', {}).copy()
                    result = {'doc_id': doc_id, 'score': similarity, 'metadata': meta}
                    if page_num is not None:
                        result['page'] = page_num
                    doc_scores[doc_id] = result
        
        results = list(doc_scores.values())
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:k]
    
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
                
            logger.info(f"Saved index with {self.index.ntotal} vectors and {len(self.doc_to_vectors)} documents")
            return True
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}", exc_info=True)
            return False
    
    def load(self):
        index_path = os.path.join(self.index_dir, "faiss.index")
        mapping_path = os.path.join(self.index_dir, "document_mapping.pkl")
        
        if not os.path.exists(index_path) or not os.path.exists(mapping_path):
            logger.warning("No saved index found")
            return False
            
        try:
            self.index = faiss.read_index(index_path, faiss.IO_FLAG_MMAP)
            
            with open(mapping_path, 'rb') as f:
                self.doc_to_vectors = pickle.load(f)
            self.id_to_doc = {}
            self.next_id = 0
            for doc_id, info in self.doc_to_vectors.items():
                for page_idx_str, page_info in info.get('pages', {}).items():
                    page_idx = int(page_idx_str)
                    start = page_info.get('start_idx', 0)
                    end = page_info.get('end_idx', start)
                    for vid in range(start, end):
                        self.id_to_doc[vid] = (doc_id, page_idx)
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