import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
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
import concurrent.futures
import aiofiles


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

        document_context_system_prompt = """
        Generate a concise document context within {self.doc_context_size} tokens by:
        
        1. Identifying key topics, themes, and structure of the document
        2. Mentioning important entities without summarizing specific data points
        3. Providing essential context for understanding the document's purpose
        4. Excluding direct quotes or paraphrases of document content
        5. Focusing only on factual information present in the document
        
        Output format: Clear, concise paragraph without headings or formatting
        """

        chunk_context_system_prompt = """
        Extract key contextual information from this document chunk in {self.chunk_context_size} tokens:
        
        1. Focus on essential facts, concepts, and entities in the current chunk
        2. Use the document summary as reference to maintain coherence
        3. Consider neighboring chunks for context but focus on the current chunk
        4. Include only factual information present in the text
        5. Avoid formatting, headings, or introductory phrases
        
        Output format: Single factual paragraph with no extraneous text
        """

        try:
            self.chunk_context_model = OllamaClient(
                hf_repo=settings.TEXT_LLM_MODEL,
                quantization=settings.TEXT_LLM_QUANTIZATION,
                system_prompt=chunk_context_system_prompt,
                temperature=float(settings.TEXT_LLM_TEMPERATURE),
                top_p=float(settings.TEXT_LLM_TOP_P),
                max_tokens=int(settings.TEXT_LLM_MAX_TOKENS),
            )

            self.document_context_model = OllamaClient(
                hf_repo=settings.TEXT_LLM_MODEL,
                quantization=settings.TEXT_LLM_QUANTIZATION,
                system_prompt=document_context_system_prompt,
                temperature=float(settings.TEXT_LLM_TEMPERATURE),
                top_p=float(settings.TEXT_LLM_TOP_P),
                max_tokens=int(settings.TEXT_LLM_MAX_TOKENS),
            )
        except (ValueError, TypeError) as e:
            logging.error(f"Error initializing Ollama client: {str(e)}")
            raise

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
            _ = await self.document_context_model.ensure_model_available()
            _ = await self.chunk_context_model.ensure_model_available()

            text = await self._extract_text_from_file(event, file_data)
            
            #TODO: Remove Document Artifacts (preprocessing the extracted text)

            chunks = await self._structured_chunking(text)

            for chunk in chunks:
                chunk['chunk_metadata'].update({key: event.get(key) for key in ['file_name', 'file_path', 'user_id', 'mime_type', 'file_size', 'description', 'document_id']})
            
            context_chunks = await self._generate_context(chunks, text)

            embedded_chunks = await self._compute_embeddings(context_chunks)

            await self._store_in_qdrant(embedded_chunks)

            logging.info(f"Successfully processed text document: {event['file_path']}")

        except Exception as e:
            logging.error(f"Failed processing text document {event['file_path']}: {str(e)}")
            await self.db.update_event_status(event['file_event_id'], 'failed', str(e))
    
    async def _extract_text_from_file(self, event, file_data):
        """Automatically detects file type and extracts text using UnstructuredLoader."""
        file_data.seek(0)
        file_content = file_data.read()

        temp_suffix = os.path.splitext(event["file_path"])[-1]
        fd, tmp_file_path = tempfile.mkstemp(suffix=temp_suffix)
        try:
            async with aiofiles.open(tmp_file_path, 'wb') as tmp_file:
                await tmp_file.write(file_content)
            
            loop = asyncio.get_event_loop()
            docs = await loop.run_in_executor(
                None, 
                lambda: UnstructuredLoader(file_path=tmp_file_path).load()
            )

            # Combine extracted text
            text = "\n\n".join([doc.page_content for doc in docs])

            if not text.strip():
                raise ValueError("Extracted text is empty")

            return text

        finally:
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
            os.close(fd)

    async def _structured_chunking(self, text):
        """Uses LangChain's RecursiveCharacterTextSplitter with enhanced metadata extraction."""
        document_chunks = self.text_splitter.create_documents([text])
        
        # Use a ThreadPoolExecutor for CPU-bound operations
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Process each chunk in parallel using threads
            futures = []
            for idx, chunk in enumerate(document_chunks):
                future = executor.submit(self._process_chunk, idx, chunk)
                futures.append(future)
            
            # Collect results
            enriched_chunks = [future.result() for future in futures]
        
        return enriched_chunks

    def _process_chunk(self, idx, chunk):
        """Process a single chunk with all metadata extraction."""
        entities = self._extract_entities(chunk.page_content)
        hierarchy = self._extract_hierarchy(chunk.page_content)
        section_type = self._detect_section_type(chunk.page_content)
        
        chunk_metadata = {
            "chunk_number": idx,
            "document_hierarchy": hierarchy,
            "entities": entities,
            "section_type": section_type
        }
        
        return {"content": chunk.page_content, "chunk_metadata": chunk_metadata}

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
        """Generates contextual embeddings for document chunks asynchronously."""
        doc_hash = await self.cache.get_hash(full_text)
        await self.cache.purge_cache()
        cached_context = await self.cache.get(doc_hash)
        if cached_context:
            logging.info("Loaded context from cache for document")
            return cached_context

        # Generate Document Summary (Only One API Call)
        doc_summary = await self.document_context_model.generate(
            f""" Focus on:
        1. Main topics and key themes
        2. Important entities and relationships
        3. Document structure and organization
        4. Essential context for retrieval
        5. Only include factual information from the text
        
        Analyze this document and create a concise summary in {self.doc_context_size} tokens:
        {full_text}

        """,
            max_tokens=self.doc_context_size
        )

        total_chunks = len(chunks)

        # Define async task for each chunk
        async def generate_chunk_context(i, chunk):
            """Generates chunk-specific context using its neighboring chunks."""
            prev_chunks = [chunks[j]["content"] for j in range(max(0, i - 2), i)]
            next_chunks = [chunks[j]["content"] for j in range(i + 1, min(total_chunks, i + 3))]

            chunk_context = await self.chunk_context_model.generate(
                f"""
                Create a contextual summary (max {self.chunk_context_size} tokens) that:
                1. Extracts key facts and entities from this chunk
                2. Connects to document summary for coherence
                3. Focuses on information relevant for search/retrieval
                4. Provides only factual information from the text
                5. Uses simple, clear language without formatting

                Document Summary: {doc_summary}
                Current Chunk ({i}/{total_chunks}): {chunk["content"]}
                Previous Context: {prev_chunks}
                Next Context: {next_chunks}
            
                """,
                max_tokens=self.chunk_context_size
            )
            chunk["chunk_metadata"]["context"] = chunk_context
            chunk["chunk_metadata"]["doc_summary"] = doc_summary
            return chunk

        # Run all chunk processing in parallel
        context_chunks = await asyncio.gather(
            *(generate_chunk_context(i, chunk) for i, chunk in enumerate(chunks))
        )

        # Store result in cache
        await self.cache.set(doc_hash, context_chunks)

        return context_chunks

    async def _compute_embeddings(self, chunks):
        """Computes hybrid embeddings (dense + sparse) for each chunk."""
        if not chunks:
            logging.warning("No chunks provided for embedding generation")
            return []
            
        tasks = [
            asyncio.gather(
                self.embedding_model.encode_dense(str(chunk["content"])),
                self.embedding_model.encode_sparse(str(chunk["content"]))
            )
            for chunk in chunks
        ]

        embeddings = await asyncio.gather(*tasks)
        for chunk, (dense_emb, sparse_emb) in zip(chunks, embeddings):
                
            chunk["dense_embedding"] = dense_emb
            chunk["sparse_embedding"] = sparse_emb

        return chunks

    async def _store_in_qdrant(self, embedded_chunks):
        """Stores processed text chunks in Qdrant with structured metadata and versioning."""
        try:
            if not embedded_chunks:
                logging.warning("No embedded chunks to store in Qdrant")
                return
                
            context_hash = await self.cache.get_hash(str(embedded_chunks))
            context_version = context_hash[:8]

            for chunk in embedded_chunks:
                chunk["chunk_metadata"]["context_version"] = context_version

            # Extract user_id safely with validation
            user_id = None
            if embedded_chunks and "chunk_metadata" in embedded_chunks[0] and "user_id" in embedded_chunks[0]["chunk_metadata"]:
                user_id = embedded_chunks[0]["chunk_metadata"]["user_id"]
            else:
                # Handle the case when user_id is not available
                logging.error("Missing user_id in chunk metadata")
                user_id = "default_user"  # Fallback value
                
            asyncio.create_task(self.qdrant.store_vectors(embedded_chunks, user_id))
            logging.info(f"Successfully stored {len(embedded_chunks)} vectors in Qdrant for user {user_id}")
            
        except Exception as e:
            logging.error(f"Failed to store vectors in Qdrant: {str(e)}")
            raise