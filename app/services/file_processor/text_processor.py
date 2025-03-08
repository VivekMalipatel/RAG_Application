import asyncio
import logging
from app.services.file_processor.entity_relation_extractor import EntityRelationExtractor
from app.core.models.model_handler import ModelRouter
from app.core.models.model_provider import Provider
from app.core.models.model_type import ModelType
from langchain_unstructured import UnstructuredLoader
from app.core.embedding.embedding_handler import EmbeddingHandler
from app.core.db_handler.document_handler import DocumentHandler
from app.core.storage_bin.minio.minio_handler import MinIOHandler
from app.core.vector_store.qdrant.qdrant_handler import QdrantHandler
from app.core.cache.redis_cache import RedisCache
from app.config import settings
import asyncio



class TextProcessor:
    """Processes text documents with hierarchical chunking, contextual augmentation, and hybrid embedding."""

    def __init__(self):
        """Initializes the TextProcessor with parameters from environment variables."""
        self.minio = MinIOHandler()
        self.db = DocumentHandler()
        self.qdrant = QdrantHandler()
        self.cache = RedisCache()

        # Fetch values from environment variables
        self.embedding_source = settings.TEXT_EMBEDDING_PROVIDER
        self.embedding_model = settings.TEXT_EMBEDDING_MODEL_NAME
        self.chunk_size = settings.TEXT_CHUNK_SIZE
        self.chunk_overlap = settings.TEXT_CHUNK_OVERLAP
        self.doc_context_size = settings.TEXT_DOCUMENT_CONTEXT_MAX_TOKENS
        self.chunk_context_size = settings.TEXT_CHUNK_CONTEXT_MAX_TOKENS

        self.embedding_model = EmbeddingHandler(
            provider=Provider(settings.TEXT_EMBEDDING_PROVIDER), 
            model_name=settings.TEXT_EMBEDDING_MODEL_NAME,  
            model_type=ModelType.TEXT_EMBEDDING
        )

        document_context_system_prompt = f"""
        You are an AI document analyst generating concise context and summary (within {self.doc_context_size} tokens) for documents to improve contextual search retrieval.
        Analyze the provided document to extract critical information:
        - Key topics, themes, and organizational structure
        - Important entities, concepts, and terminology
        - Document purpose, audience, and domain context
        - Document's scope, time frame, and contextual setting
        - Document type and format information

        Focus only on factual, search-enhancing information. Ignore file artifacts (newlines, etc.) and avoid phrases like "This document discusses."
        Respond with a single, concise paragraph that would help retrieve this document during semantic search.
        """

        chunk_context_system_prompt = f"""
        You are an AI document analyst generating concise context (within {self.chunk_context_size} tokens) for text chunks to improve search retrieval.
        Analyze the provided chunk within its document context to identify:
        - Core topic/concept and its significance
        - Connection to surrounding content
        - Relationship to main document themes
        - Key entities and concepts
        - Position in document structure
        - Technical terminology introduced
        - Contribution to overall document purpose

        Respond with a single, concise paragraph of contextual information that would help retrieve this chunk during semantic search.
        """
        try:
            self.chunk_context_model = ModelRouter(
                provider=Provider(settings.TEXT_CONTEXT_LLM_PROVIDER),
                model_name=settings.TEXT_CONTEXT_LLM_MODEL_NAME,
                model_quantization=settings.TEXT_CONTEXT_LLM_QUANTIZATION,
                model_type=ModelType.TEXT_GENERATION,
                system_prompt=chunk_context_system_prompt,
                temperature=settings.TEXT_CONTEXT_LLM_TEMPERATURE,
                top_p=settings.TEXT_CONTEXT_LLM_TOP_P,
                max_tokens=settings.TEXT_DOCUMENT_CONTEXT_MAX_TOKENS,
            )
            
            self.document_context_model = ModelRouter(
                provider=Provider(settings.TEXT_CONTEXT_LLM_PROVIDER),  
                model_name=settings.TEXT_CONTEXT_LLM_MODEL_NAME,
                model_quantization=settings.TEXT_CONTEXT_LLM_QUANTIZATION,
                model_type=ModelType.TEXT_GENERATION,
                system_prompt=document_context_system_prompt,
                temperature=settings.TEXT_CONTEXT_LLM_TEMPERATURE,
                top_p=settings.TEXT_CONTEXT_LLM_TOP_P,
                max_tokens=settings.TEXT_CHUNK_CONTEXT_MAX_TOKENS,
            )
            """
            self.document_context_model = ModelRouter(
                provider=Provider("openai"),  
                model_name="deepseek-chat",
                model_quantization=settings.TEXT_CONTEXT_LLM_QUANTIZATION,
                model_type=ModelType.TEXT_GENERATION,
                system_prompt=document_context_system_prompt,
                temperature=settings.TEXT_CONTEXT_LLM_TEMPERATURE,
                top_p=settings.TEXT_CONTEXT_LLM_TOP_P,
                max_tokens=settings.TEXT_CHUNK_CONTEXT_MAX_TOKENS,
            )
            """
        except (ValueError, TypeError) as e:
            logging.error(f"Error initializing Ollama client: {str(e)}")
            raise

        self.entity_relation_extractor = EntityRelationExtractor()

    async def process(self, event, file_data):
        """Main processing function for text documents."""
        try:

            chunks, document_text = await self._extract_text_from_file(event, file_data)

            chunks = await self._generate_context(chunks, document_text)

            chunks = await self.entity_relation_extractor.extract_entities_and_relationships(chunks)

            chunks = await self._compute_embeddings(chunks)

            await self._store_in_qdrant(chunks)

            logging.info(f"Successfully processed text document: {event['file_path']}")

        except Exception as e:
            logging.error(f"Failed processing text document {event['file_path']}: {str(e)}")
            await self.db.update_event_status(event['file_event_id'], 'failed', str(e))
    
    async def _extract_text_from_file(self, event, file_data):
        """Extracts text chunks from a file using UnstructuredLoader and maps to our chunk schema."""
        try:
            file_data.seek(0)

            # Load document and extract structured chunks with metadata
            loader = UnstructuredLoader(
                file=file_data,
                metadata_filename=event['file_name'],
                strategy='hi_res',
                chunking_strategy="by_title",
                max_characters=int(self.chunk_size * 75 / 100),
                overlap = self.chunk_overlap,
                include_orig_elements=False,
            )

            docs = await loader.aload()

            if not docs:
                raise ValueError("No extracted content from the document")

            # Convert extracted documents into our chunk format
            chunks,document_text = self._convert_unstructured_chunks(docs, event)

            logging.info(f"Successfully extracted and structured {len(chunks)} chunks from {event['file_name']}")
            return chunks, document_text

        except Exception as e:
            logging.error(f"Error extracting text from {event['file_name']}: {str(e)}")
            raise

    def _convert_unstructured_chunks(self, docs, event):
        """Converts UnstructuredLoader output into our structured chunk format with additional metadata."""
        structured_chunks = []
        document_text = ""

        for idx, doc in enumerate(docs):
            document_text += doc.page_content

            chunk_metadata = {
                "chunk_number": idx,
                "document_id": event["document_id"],
                "user_id": event["user_id"],
                "file_name": event["file_name"],
                "file_path": event["file_path"],
                "file_size": event["file_size"],
                "description": event["description"],
                "page_number": doc.metadata.get("page_number", None),
                "mime_type": doc.metadata.get("filetype", None),
                "languages": doc.metadata.get("languages", None),
                "category": doc.metadata.get("category", None),
                "element_id": doc.metadata.get("element_id", None),
                "parent_id": doc.metadata.get("parent_id", None),
                "is_continuation": doc.metadata.get("is_continuation", False),
            }

            if chunk_metadata["parent_id"] is not None:
                print(f"Parent ID: {chunk_metadata['parent_id']}")

            structured_chunks.append({
                "content": doc.page_content,
                "chunk_metadata": chunk_metadata
            })

        return structured_chunks, document_text

    async def _generate_context(self, chunks, full_text):
        """Generates contextual embeddings for document chunks asynchronously."""
        doc_hash = await self.cache.get_hash(full_text)
        #await self.cache.purge_cache()
        cached_context = await self.cache.get(doc_hash)
        if cached_context:
            logging.info("Loaded context from cache for document")
            return cached_context

        # Generate Document Summary (Only One API Call)
        doc_summary = await self.document_context_model.generate_text(
            prompt=f"""
                Document (Extracted from a RAW Byte Data of a file):
                {full_text} 
                Context:
                """,
            max_tokens=self.doc_context_size
        )

        total_chunks = len(chunks)

        # Define async task for each chunk
        async def generate_chunk_context(i, chunk):
            """Generates chunk-specific context using its neighboring chunks."""
            if settings.TEXT_CONTEXT_LLM_PROVIDER == Provider.OPENAI:
                chunk_context = await self.chunk_context_model.generate_text(
                    prompt=f"""
                        Document: 
                        {full_text}
                        
                        Current Chunk ({i}/{total_chunks}): 
                        
                        {chunk}
                        
                        Context:
                        """,
                    max_tokens=self.chunk_context_size
                )
            else:
                prev_chunks = [chunks[j]["content"] for j in range(max(0, i - 2), i)]
                next_chunks = [chunks[j]["content"] for j in range(i + 1, min(total_chunks, i + 3))]

                chunk_context = await self.chunk_context_model.generate_text(
                    prompt=f"""
                        Document: 
                        {doc_summary}
                        
                        Current Chunk ({i}/{total_chunks}): 
                        
                        {chunk}
                        
                        Previous Two Chunks: 
                        {prev_chunks}
                        
                        Next Two Chunks: 
                        {next_chunks}
                        
                        Context:
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
        
        #TODO: Implement batching for large number of chunks

        tasks = [
            asyncio.gather(
                self.embedding_model.encode_dense(str(chunk["content"])),
                self.embedding_model.encode_sparse(str(chunk["content"]))
            )
            for chunk in chunks
        ]

        embeddings = await asyncio.gather(*tasks)
        for chunk, (dense_emb, sparse_emb) in zip(chunks, embeddings):
                
            chunk["dense_embedding"] = dense_emb[0]
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