import os
import numpy as np
import faiss
import pickle
import logging
import time
import asyncio
from typing import List, Dict, Any
from app.core.storage.s3_handler import S3Handler
from app.config import settings

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, index_dir: str = "storage/faiss_indexes", embedding_dim: int = 2048, use_ivf: bool = False, use_pq: bool = False, use_gpu: bool = False):
        self.index_dir = index_dir
        self.embedding_dim = embedding_dim
        self.use_ivf = use_ivf
        self.use_pq = use_pq
        self.use_gpu = use_gpu
        self.index = None
        self.doc_to_vectors: Dict[str, Dict] = {}
        self.id_to_doc: Dict[int, str] = {}
        self.next_id = 0
        self.s3_handler = S3Handler()
        self.s3_prefix = f"{settings.S3_BACKUP_PREFIX}faiss_indexes/"
        os.makedirs(self.index_dir, exist_ok=True)
        logger.info(f"VectorStore initialized: dim={embedding_dim}, ivf={use_ivf}, pq={use_pq}, gpu={use_gpu}")
        self.warmup_samples: List[np.ndarray] = []
        self.warmup_limit = 10000
        self.min_warmup = 100

    def _build_hnsw(self):
        M, efC, efS = 32, 200, 200
        idx = faiss.IndexHNSWFlat(self.embedding_dim, M, faiss.METRIC_INNER_PRODUCT)
        idx.hnsw.efConstruction = efC
        idx.hnsw.efSearch = efS
        return idx

    def _build_ivf(self, total_vectors: int):
        nlist = max(1, int(4 * np.sqrt(total_vectors)))
        quant = faiss.IndexFlatIP(self.embedding_dim)
        if self.use_pq:
            m = 8
            idx = faiss.IndexIVFPQ(quant, self.embedding_dim, nlist, m, 8)
        else:
            idx = faiss.IndexIVFFlat(quant, self.embedding_dim, nlist, faiss.METRIC_INNER_PRODUCT)
        idx.nprobe = max(1, int(0.1 * nlist))
        return idx

    def initialize_index(self, total_vectors: int = 0):
        if self.use_ivf:
            base = self._build_ivf(total_vectors)
        else:
            base = self._build_hnsw()
        self.index = faiss.IndexIDMap(base)
        if self.use_ivf and not self.index.is_trained:
            total_warm = sum(s.shape[0] for s in self.warmup_samples)
            if total_warm >= self.min_warmup:
                sample = np.vstack(self.warmup_samples)[:self.warmup_limit]
                self.index.train(sample)
        if self.use_gpu and faiss.get_num_gpus() > 0:
            opts = faiss.GpuClonerOptions()
            opts.shard = True
            self.index = faiss.index_cpu_to_all_gpus(self.index, opts)
            res = faiss.StandardGpuResources()
            res.setTempMemory(1024 * 1024 * 1024)
            logger.info("Moved index to GPU with reserved temp memory")

    def add_document(self, doc_id: str, embeddings: Any, metadata: Dict[str, Any] = None) -> int:
        arr = np.array(embeddings, dtype='float32')
        assert arr.ndim == 2 and arr.shape[1] == self.embedding_dim
        faiss.normalize_L2(arr)
        if self.use_ivf:
            total_warm = sum(s.shape[0] for s in self.warmup_samples)
            if total_warm < self.warmup_limit:
                needed = self.warmup_limit - total_warm
                if needed > 0:
                    self.warmup_samples.append(arr[:needed])
        if self.index is None:
            self.initialize_index(total_vectors=arr.shape[0])
        if self.use_ivf and not self.index.is_trained:
            sample = arr[np.random.choice(arr.shape[0], size=min(arr.shape[0], 10000), replace=False)]
            self.index.train(sample)
        ids = np.arange(arr.shape[0], dtype='int64') + self.next_id
        self.index.add_with_ids(arr, ids)
        for i, vid in enumerate(ids):
            self.id_to_doc[int(vid)] = (doc_id, i)
        count = arr.shape[0]
        pages = {i: int(vid) for i, vid in enumerate(ids)}
        self.doc_to_vectors[doc_id] = {
            'count': count,
            'metadata': metadata or {},
            'raw_embeddings': embeddings,
            'pages': pages
        }
        self.next_id += count
        logger.info(f"Added document {doc_id} with {count} vectors")
        
        if self.index.ntotal % 100 == 0:
            asyncio.create_task(self._save_to_s3())
        
        if self.use_ivf and self.should_rebuild():
            logger.info("Re-training IVF quantizer due to high churn")
            self.rebuild_index(use_gpu=self.use_gpu)
        return count

    def search(self, query_vectors: Any, k: int = 10, nprobe: int = None, efSearch: int = None) -> List[Dict[str, Any]]:
        t0 = time.time()
        arr = np.array(query_vectors, dtype='float32')
        assert arr.ndim == 2 and arr.shape[1] == self.embedding_dim
        faiss.normalize_L2(arr)
        if self.use_ivf and nprobe:
            self.index.nprobe = nprobe
        if not self.use_ivf and efSearch:
            try:
                self.index.hnsw.efSearch = efSearch
            except AttributeError:
                pass
        total = getattr(self.index, 'ntotal', 0)
        if total == 0:
            return []
        distances, indices = self.index.search(arr, k)
        results = []
        for sims, vids in zip(distances, indices):
            hits = []
            for sim, vid in zip(sims, vids):
                if vid < 0:
                    break
                doc_pg = self.id_to_doc.get(int(vid))
                if doc_pg is None:
                    continue
                doc_id, page_idx = doc_pg
                hits.append({'doc_id': doc_id, 'page_idx': page_idx, 'score': float(sim), 'metadata': self.doc_to_vectors[doc_id]['metadata']})
            results.append(hits)
        t1 = time.time()
        logger.info(f"Search latency k={k}: {t1-t0:.4f}s")
        return results

    def save(self) -> bool:
        if self.index is None:
            return False
        os.makedirs(self.index_dir, exist_ok=True)
        idx = faiss.index_gpu_to_cpu(self.index) if self.use_gpu else self.index
        
        idx_path = os.path.join(self.index_dir, 'faiss.index')
        map_path = os.path.join(self.index_dir, 'mapping.pkl')
        
        faiss.write_index(idx, idx_path)
        with open(map_path, 'wb') as f:
            pickle.dump({'doc_to_vectors': self.doc_to_vectors, 'id_to_doc': self.id_to_doc}, f)
        
        logger.info(f"Saved index with {idx.ntotal} vectors to local storage")
        
        asyncio.create_task(self._save_to_s3())
        
        return True

    async def _save_to_s3(self):
        try:
            idx_path = os.path.join(self.index_dir, 'faiss.index')
            map_path = os.path.join(self.index_dir, 'mapping.pkl')
            
            if os.path.exists(idx_path):
                await self.s3_handler.upload_file(idx_path, f"{self.s3_prefix}faiss.index")
            
            if os.path.exists(map_path):
                await self.s3_handler.upload_file(map_path, f"{self.s3_prefix}mapping.pkl")
            
            logger.info("Successfully synced FAISS index to S3")
            
        except Exception as e:
            logger.error(f"Error syncing FAISS index to S3: {str(e)}")

    def load(self) -> bool:
        return asyncio.run(self._load_async())

    async def _load_async(self) -> bool:
        idx_path = os.path.join(self.index_dir, 'faiss.index')
        map_path = os.path.join(self.index_dir, 'mapping.pkl')
        
        local_exists = os.path.exists(idx_path) and os.path.exists(map_path)
        
        if not local_exists:
            logger.info("No local index found, checking S3...")
            s3_loaded = await self._load_from_s3()
            if not s3_loaded:
                logger.info("No saved index in S3; initializing new one")
                self.initialize_index()
                return True
        
        if os.path.exists(idx_path) and os.path.exists(map_path):
            try:
                self.index = faiss.read_index(idx_path)
                with open(map_path, 'rb') as f:
                    mapping = pickle.load(f)
                    self.doc_to_vectors = mapping.get('doc_to_vectors', {})
                    self.id_to_doc = mapping.get('id_to_doc', {})
                self.next_id = max(self.id_to_doc.keys(), default=-1) + 1
                logger.info(f"Loaded index with {self.index.ntotal} vectors from local storage")
                return True
            except Exception as e:
                logger.error(f"Error loading index from local storage: {str(e)}")
                logger.info("Attempting to load from S3...")
                return await self._load_from_s3()
        
        return False

    async def _load_from_s3(self) -> bool:
        try:
            idx_s3_key = f"{self.s3_prefix}faiss.index"
            map_s3_key = f"{self.s3_prefix}mapping.pkl"
            
            idx_exists = await self.s3_handler.object_exists(idx_s3_key)
            map_exists = await self.s3_handler.object_exists(map_s3_key)
            
            if not (idx_exists and map_exists):
                logger.info("FAISS index files not found in S3")
                return False
            
            idx_path = os.path.join(self.index_dir, 'faiss.index')
            map_path = os.path.join(self.index_dir, 'mapping.pkl')
            
            idx_downloaded = await self.s3_handler.download_file(idx_s3_key, idx_path)
            map_downloaded = await self.s3_handler.download_file(map_s3_key, map_path)
            
            if idx_downloaded and map_downloaded:
                self.index = faiss.read_index(idx_path)
                with open(map_path, 'rb') as f:
                    mapping = pickle.load(f)
                    self.doc_to_vectors = mapping.get('doc_to_vectors', {})
                    self.id_to_doc = mapping.get('id_to_doc', {})
                self.next_id = max(self.id_to_doc.keys(), default=-1) + 1
                logger.info(f"Loaded index with {self.index.ntotal} vectors from S3")
                return True
            else:
                logger.error("Failed to download FAISS index files from S3")
                return False
                
        except Exception as e:
            logger.error(f"Error loading FAISS index from S3: {str(e)}")
            return False

    def get_stats(self):
        if self.index is None:
            return {"status": "not_initialized"}
        stats = {
            "total_vectors": self.index.ntotal,
            "total_documents": len(self.doc_to_vectors),
            "embedding_dimension": self.embedding_dim,
            "index_type": type(self.index).__name__
        }
        if hasattr(self.index, 'nprobe'):
            stats.update({"nlist": getattr(self.index, 'nlist', None), "nprobe": self.index.nprobe})
        if hasattr(self.index, 'hnsw'):
            stats.update({"M": self.index.hnsw.M, "efConstruction": self.index.hnsw.efConstruction, "efSearch": self.index.hnsw.efSearch})
        return stats

    def remove_document(self, doc_id: str) -> bool:
        if doc_id not in self.doc_to_vectors:
            logger.warning(f"Document {doc_id} not found")
            return False
        vids = [vid for vid, did in self.id_to_doc.items() if did == doc_id]
        for vid in vids:
            self.id_to_doc.pop(vid, None)
        self.doc_to_vectors.pop(doc_id, None)
        logger.info(f"Removed document {doc_id} with {len(vids)} vectors")
        return True

    def search_batch(self, queries: Any, k: int = 10, nprobe: int = None, efSearch: int = None) -> List[List[Dict[str, Any]]]:
        t0 = time.time()
        arr = np.array(queries, dtype='float32')
        assert arr.ndim == 2 and arr.shape[1] == self.embedding_dim
        faiss.normalize_L2(arr)
        if self.use_ivf and nprobe:
            self.index.nprobe = nprobe
        if not self.use_ivf and efSearch:
            try:
                self.index.hnsw.efSearch = efSearch
            except AttributeError:
                pass
        total = getattr(self.index, 'ntotal', 0)
        if total == 0:
            return [[] for _ in range(arr.shape[0])]
        distances, indices = self.index.search(arr, k)
        batch_results = []
        for sims, vids in zip(distances, indices):
            hits = []
            for sim, vid in zip(sims, vids):
                if vid < 0:
                    break
                doc_pg = self.id_to_doc.get(int(vid))
                if doc_pg is None:
                    continue
                doc_id, page_idx = doc_pg
                hits.append({'doc_id': doc_id, 'page_idx': page_idx, 'score': float(sim), 'metadata': self.doc_to_vectors[doc_id]['metadata']})
            batch_results.append(hits)
        t1 = time.time()
        logger.info(f"Batch search latency k={k}: {t1-t0:.4f}s")
        return batch_results

    def rebuild_index(self, use_gpu: bool = False) -> bool:
        if not self.doc_to_vectors:
            logger.warning("No documents to rebuild index")
            return False
        total_vecs = sum(info['count'] for info in self.doc_to_vectors.values())
        self.index = None
        self.next_id = 0
        self.initialize_index(total_vectors=total_vecs)
        self.id_to_doc.clear()
        for doc_id, info in self.doc_to_vectors.items():
            arr = np.array(info['raw_embeddings'], dtype='float32')
            faiss.normalize_L2(arr)
            ids = np.arange(arr.shape[0], dtype='int64') + self.next_id
            self.index.add_with_ids(arr, ids)
            for i, vid in enumerate(ids):
                self.id_to_doc[int(vid)] = (doc_id, i)
            self.next_id += arr.shape[0]
        logger.info(f"Rebuilt index with {self.index.ntotal} vectors")
        return True

    def should_rebuild(self, threshold: float = 0.2) -> bool:
        total = getattr(self.index, 'ntotal', 0)
        if total == 0:
            return False
        active = sum(info['count'] for info in self.doc_to_vectors.values())
        deleted = total - active
        return (deleted / total) >= threshold

    async def shutdown(self):
        logger.info("Shutting down VectorStore and syncing to S3...")
        self.save()
        await self._save_to_s3()
        logger.info("VectorStore shutdown complete")