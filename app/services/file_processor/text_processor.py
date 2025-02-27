import asyncio
import logging
from datetime import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_unstructured import UnstructuredLoader
import spacy
from app.core.embedding.embedding_handler import EmbeddingHandler
from app.core.db_handler.document_handler import DocumentHandler
from app.core.storage_bin.minio.minio_handler import MinIOHandler
from app.core.vector_store.qdrant.qdrant_handler import QdrantHandler
from app.core.models.ollama.ollama import OllamaClient
from app.core.cache.redis_cache import RedisCache
from app.config import settings
from spacy.cli import download
import os
import tempfile

class TextProcessor:
    """Processes text documents with hierarchical chunking, contextual augmentation, and hybrid embedding."""

    def __init__(self):
        """Initializes the TextProcessor with parameters from environment variables."""
        self.minio = MinIOHandler()
        self.db = DocumentHandler()
        self.qdrant = QdrantHandler()
        self.cache = RedisCache()

        # Fetch values from environment variables
        self.embedding_source = settings.TEXT_EMBEDDING_SOURCE
        self.embedding_model = settings.TEXT_EMBEDDING_MODEL
        self.chunk_size = int(settings.TEXT_CHUNK_SIZE)
        self.chunk_overlap = int(settings.TEXT_CHUNK_OVERLAP)
        self.doc_context_size = int(settings.TEXT_DOC_CONTEXT_SIZE)
        self.chunk_context_size = int(settings.TEXT_CHUNK_CONTEXT_SIZE)

        self.embedding_model = EmbeddingHandler(
            model_source=self.embedding_source, model_name=self.embedding_model, model_type="text"
        )

        system_prompt = """
        You are an advanced assistant optimized for document context extraction. Your task is to generate concise, factually accurate, and relevant contextual summaries for text chunks in a Retrieval-Augmented Generation (RAG) pipeline.

        ### Instructions:
        - Extract **key contextual details** from each document chunk.
        - Utilize the **document summary** as guidance to maintain coherence.
        - Ensure responses are **factually accurate** and **contextually relevant**.
        - Adhere strictly to the **max token limit** for both chunk-based and document-level contexts.
        - Avoid unnecessary repetition and focus on **essential information**.

        Your goal is to enhance retrieval by providing **well-structured, meaningful context** for each chunk.
        """

        self.ollama = OllamaClient(
            hf_repo=settings.TEXT_LLM_MODEL,
            quantization=settings.TEXT_LLM_QUANTIZATION,
            system_prompt=system_prompt,
            temperature=float(settings.TEXT_LLM_TEMPERATURE),
            top_p=float(settings.TEXT_LLM_TOP_P),
            max_tokens=int(settings.TEXT_LLM_MAX_TOKENS),
        )

        # LangChain RecursiveCharacterTextSplitter with hierarchy handling
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=[
                "\n\n", "\nChapter", "\n##", "\n###", "\n- ", "\n* ", "\n"
            ]
        )

        # Load SpaCy NLP for Named Entity Recognition
        self.ner_model = self._initialize_spacy_model('en_core_web_sm')
    
    def _initialize_spacy_model(self, model_name, max_retries=3):
        """Initialize spaCy model with download handling and retries."""
        for attempt in range(max_retries):
            try:
                return spacy.load(model_name)
            except OSError:
                logging.info(f"Downloading spaCy model: {model_name}")
                try:
                    download(model_name)
                    return spacy.load(model_name)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logging.error(f"Failed to download spaCy model after {max_retries} attempts: {str(e)}")
                        raise RuntimeError(f"Could not initialize NER model: {str(e)}")
                    logging.warning(f"Attempt {attempt + 1} failed, retrying...")
                    continue

    async def process(self, event, file_data):
        """Main processing function for text documents."""
        try:

            _ = await self.ollama.ensure_model_available()

            text = await self._extract_text_from_file(event, file_data)

            # Step 1: Hierarchical Chunking with Metadata
            chunks = await self._structured_chunking(text)

            # Step 2: Context Generation (Hybrid Extractive + Abstractive)
            context_task = self._generate_context(chunks, text)

            # Step 3: Compute Embeddings (Parallelized)
            context = await context_task
            doc_summary = context["doc_summary"]
            context_chunks = context["context_chunks"]
            enriched_chunks = []
            for original_chunk, context in zip(chunks, context_chunks):
                combined_content = {
                    "original_content": original_chunk["content"],
                    "context": context,
                    "doc_summary": doc_summary,
                    "metadata": original_chunk["metadata"]
                }
                enriched_chunks.append(combined_content)
                
            embedding_task = self._compute_embeddings(enriched_chunks)

            # Step 4: Store in Qdrant
            embedded_chunks = await embedding_task
            store_task = self._store_in_qdrant(event, embedded_chunks)

            await store_task
            logging.info(f"Successfully processed text document: {event['path']}")

        except Exception as e:
            logging.error(f"Failed processing text document {event['path']}: {str(e)}")
            await self.db.update_event_status(event['event_id'], 'failed', str(e))
    
    async def _extract_text_from_file(self, event, file_data):
        """Automatically detects file type and extracts text using UnstructuredLoader."""
        file_data.seek(0)
        file_content = file_data.read()

        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(event["path"])[-1]) as tmp_file:
            tmp_file.write(file_content)
            tmp_file_path = tmp_file.name

        try:
            # Load file using UnstructuredLoader
            loader = UnstructuredLoader(file_path=tmp_file_path)
            docs = await loader.aload()  # Async loading

            # Combine extracted text
            text = "\n\n".join([doc.page_content for doc in docs])

            if not text.strip():
                raise ValueError("Extracted text is empty")

            return text

        finally:
            os.unlink(tmp_file_path)

    async def _structured_chunking(self, text):
        """Uses LangChain's RecursiveCharacterTextSplitter with enhanced metadata extraction."""
        document_chunks = self.text_splitter.create_documents([text])
        enriched_chunks = []

        for idx, chunk in enumerate(document_chunks):
            # Extract entities using NLP
            entities = self._extract_entities(chunk.page_content)

            # Extract Hierarchy (Improved)
            hierarchy = self._extract_hierarchy(chunk.page_content)

            # Enrich metadata
            metadata = {
                "chunk_number": idx,  # Track chunk position
                "document_hierarchy": hierarchy,
                "entities": entities,
                "section_type": self._detect_section_type(chunk.page_content)
            }
            enriched_chunks.append({"content": chunk.page_content, "metadata": metadata})

        return enriched_chunks

    def _extract_hierarchy(self, text):
        """Extracts and preserves document hierarchy based on headers."""
        lines = text.split("\n")
        hierarchy = []
        for line in lines:
            if line.strip().startswith(("Chapter", "##", "###")):
                hierarchy.append(line.strip())
        return hierarchy[-3:]  # Retain last 3 levels

    def _extract_entities(self, text):
        """Extracts Named Entities for Light RAG Readiness."""
        doc = self.ner_model(text)
        return [{"text": ent.text, "label": ent.label_} for ent in doc.ents]

    def _detect_section_type(self, text):
        """Classifies section type for metadata tagging."""
        if "Table of Contents" in text:
            return "TOC"
        elif "References" in text:
            return "References"
        elif "Appendix" in text:
            return "Appendix"
        else:
            return "Body"

    async def _generate_context(self, chunks, full_text):
        """Generates contextual embeddings for document chunks."""
        doc_hash = await self.cache.get_hash(full_text)
        cached_context = await self.cache.get(doc_hash)
        if cached_context:
            logging.info("Loaded context from cache for document")
            return cached_context

        # Generate Document Summary
        doc_summary = await self.ollama.generate(
            f"Create a concise summary (max {self.doc_context_size} tokens) of this document:\n{full_text}",
            max_tokens=self.doc_context_size
        )

        context_chunks = []
        total_chunks = len(chunks)

        for i, chunk in enumerate(chunks):
            # Get surrounding chunks (i-2, i-1, i+1, i+2)
            prev_chunks = [chunks[j]["content"] for j in range(max(0, i - 2), i)]
            next_chunks = [chunks[j]["content"] for j in range(i + 1, min(total_chunks, i + 3))]

            # Generate Context-Aware Summary
            chunk_context = await self.ollama.generate(
                f"""
                Here is the Document Summary from which the below data is chunked from : {doc_summary}
                Here is the chunk : {chunk["content"]}
                Here are the preceeding chunks : {prev_chunks}
                Here are the following chunks : {next_chunks}
                Now generate the Chunk Context (max {self.chunk_context_size} tokens) by exracting key contextual information for retrieval leveraging the document summary, preceeding and following chunks. 
                """,
                max_tokens=self.chunk_context_size
            )

            # Store final metadata + context
            context_chunks.append({
                "content": chunk["content"],
                "chunk_number": chunk["metadata"]["chunk_number"],
                "context": chunk_context,
                "document_hierarchy": chunk["metadata"]["document_hierarchy"],
                "entities": chunk["metadata"]["entities"],
                "section_type": chunk["metadata"]["section_type"]
            })

        return_data = {
            "doc_summary": doc_summary,
            "context_chunks": context_chunks
        }

        await self.cache.set(doc_hash, return_data)
        return return_data

    async def _compute_embeddings(self, context_chunks):
        """Computes hybrid embeddings (dense + sparse) for each chunk, integrating multi-modal compatibility."""
        embedded_chunks = []

        for chunk in context_chunks:
            # Merge all context for embedding
            chunk_content = (
                f"Document Summary: {chunk['doc_summary']}\n"
                f"Chunk Context: {chunk['context']}\n"
                f"Entities: {chunk['entities']}\n"
                f"Document Hierarchy: {chunk['document_hierarchy']}\n"
                f"Section Type: {chunk['section_type']}\n"
                f"Chunk Content: {chunk['content']}"
            )

            # Compute embeddings
            dense_embedding = await self.embedding_model.encode_dense(chunk_content)
            sparse_embedding = await self.embedding_model.encode_sparse(chunk_content)

            # Store chunk embedding with metadata
            embedded_chunks.append({
                "chunk_number": chunk["chunk_number"],  # Ensure chunk indexing
                "dense_embedding": dense_embedding,
                "sparse_embedding": sparse_embedding,
                "document_hierarchy": chunk["document_hierarchy"],
                "entities": chunk["entities"],
                "section_type": chunk["section_type"],
                "content": chunk["content"]
            })

        return embedded_chunks

    async def _store_in_qdrant(self, event, embedded_chunks):
        """Stores processed text chunks in Qdrant with structured metadata and versioning."""

        context_hash = await self.cache.get_hash(str(embedded_chunks))
        context_version = context_hash[:8]

        metadata = {
            "document_id": event["event_id"],
            "user_id": event["user_id"],
            "timestamp": str(datetime.utcnow()),
            "file_path": event["path"],
            "context_version": context_version
        }

        # Prepare final chunk structure
        formatted_chunks = [
            {
                "chunk_number": chunk["chunk_number"],
                "dense_embedding": chunk["dense_embedding"],
                "sparse_embedding": chunk["sparse_embedding"],
                "metadata": {
                    **metadata,
                    "document_hierarchy": chunk["document_hierarchy"],
                    "entities": chunk["entities"],
                    "section_type": chunk["section_type"],
                    "content": chunk["content"]
                }
            }
            for chunk in embedded_chunks
        ]

        await self.qdrant.store_vectors(formatted_chunks, metadata)