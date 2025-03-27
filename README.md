# OmniRAG Application

OmniRAG is an advanced Retrieval-Augmented Generation system implementing a hybrid dense-sparse vector architecture with multi-modal LLM integration. The system combines vector search (Qdrant), knowledge graph capabilities (Neo4j), and LangGraph-based agent workflows to deliver context-aware responses. Technical highlights include asynchronous embedding generation with Redis-backed caching, intelligent document chunking with cross-chunk entity resolution, reciprocal rank fusion for result combination, and dynamic context window optimization through the LightRAG methodology. The application supports both self-hosted LLMs via Ollama and cloud models through a unified model router abstraction.

## Features

- **Document Processing**: Upload and process various file formats (PDF, DOCX, TXT, etc.)
- **Vector Database Integration**: Efficient storage and retrieval of document embeddings
- **Contextual Answering**: Get answers based on the content of your documents
- **Conversational History**: Maintain context through multi-turn conversations
- **API Support**: Access all functionality through a RESTful API
- **Multi-Agent Workflows**: LangGraph-based agent orchestration for complex queries
- **Entity Extraction**: Structured information extraction and relationship mapping
- **Knowledge Graph Integration**: Store and query entity relationships for enhanced context
- **Hybrid Search**: Combination of vector and keyword-based search methodologies
- **LightRAG**: Optimized retrieval pipeline with efficient caching and token reduction techniques

## Architecture

OmniRAG uses a FastAPI backend with a modular monolithic architecture that integrates:

1. **Document Processing Pipeline**:
   - Chunks documents with intelligent continuation handling
   - Performs entity and relationship extraction across chunks
   - Generates embeddings with state-of-the-art models

2. **Search Orchestration**:
   - Employs a multi-step workflow for query analysis
   - Performs preliminary and optimized searches
   - Verifies result quality and follows up when necessary
   - Ranks and compiles results from multiple sources

3. **Agent Framework**:
   - Based on LangGraph for structured agent workflows
   - Utilizes the BaseAgent abstraction for modular agent design
   - Supports multi-agent collaboration for complex queries

4. **Storage Layer**:
   - Qdrant for vector embeddings storage and similarity search
   - Neo4j for entity relationships and knowledge graph queries
   - Redis for caching and performance optimization
   - Minio for object storage of original documents

## Technologies & Methodologies

### Core Technology Stack
- **Backend Framework**: FastAPI (Python) with asynchronous request handling
- **LLM Integration**: 
  - Ollama for self-hosted models (LLaMA, Mistral, Falcon)
  - Hugging Face for specialized embedding and classification models
  - OpenAI compatibility layer for API integrations
- **Vector Database**: Qdrant with both dense and sparse vector support
- **Graph Database**: Neo4j with vector indexing capabilities
- **Caching**: Redis with semantic deduplication and tiered caching strategies
- **Object Storage**: Minio S3-compatible service for document persistence
- **Task Queue**: Celery for background processing and task distribution
- **Embedding Models**:
  - Sentence transformers for dense embeddings
  - BM25 for sparse lexical embeddings (via Qdrant/bm25)
  - Custom multi-modal embedding pipeline for mixed content types

### Advanced Methodologies
- **RAG Techniques**:
  - Hybrid Dense-Sparse Retrieval with fusion algorithms
  - Self-querying retrieval with structured metadata filtering
  - Multi-query expansion for improved recall
  - Contextual compression for token optimization
  - LightRAG for efficient retrieval with reduced token usage

- **Knowledge Processing**:
  - Cross-document coreference resolution
  - Two-pass relationship extraction for improved accuracy
  - Entity disambiguation using graph-based validation
  - Metadata-enriched entity extraction

- **Agent Architecture**:
  - LangGraph-based orchestration with state management
  - Multi-agent collaboration with specialized roles
  - Recursive agent workflows for complex query decomposition
  - Agentic retrieval optimization with dynamic search refinement

- **Performance Optimizations**:
  - Multi-tier caching with semantic fingerprinting
  - Asynchronous parallel processing for embedding generation
  - Batched vector operations for throughput maximization
  - Incremental indexing for large document collections

## Implementation Details

### Vector Search Infrastructure
- **Dual Vector Strategy**: Simultaneous dense and sparse vector encoding
- **Vector Indexing**: HNSW algorithm in Qdrant for approximate nearest neighbors
- **Filtering**: Payload-based filtering with composite filtering conditions
- **Scoring**: Reciprocal rank fusion algorithm for result combination

### Knowledge Graph Implementation
- **Entity Registry**: Cross-document entity resolution and linking
- **Vector-Enabled Nodes**: Neo4j vector indexes for semantic similarity in graph
- **Relationship Embeddings**: Vectorized relationship properties for semantic querying
- **Traversal Algorithms**: Custom graph traversal for knowledge exploration

### Embedding Pipeline
- **Caching Strategy**: Redis-based embedding cache with hash-based lookup
- **Provider Abstraction**: Unified interface for multiple embedding sources
- **Error Handling**: Graceful degradation with fallback mechanisms
- **Embedding Fusion**: Techniques for combining different embedding types

### Document Processing
- **Chunking Strategies**: Intelligent chunk boundaries with semantic awareness
- **Overlap Management**: Controlled token overlap with duplicate removal
- **Metadata Extraction**: Automated metadata extraction from document properties
- **Format Support**: Handlers for PDF, DOCX, TXT, HTML, and markdown formats

## Prerequisites

- Docker and Docker Compose
- Python 3.9+
- Sufficient disk space for document storage and embeddings

## Setup

### Using Docker (Recommended)

```bash
# Docker activation
docker compose up -d

# Docker deactivation
docker compose down

# Check Docker status
docker ps
```

### Manual Setup

1. Create a virtual environment:
```bash
python3 -m venv .rag
source .rag/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start the application:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Usage

Once the application is running, access the web interface at:
```
http://localhost:8000
```

API documentation is available at:
```
http://localhost:8000/docs
```

## Advanced Features

### Model Router Architecture

The Model Router is a critical component of OmniRAG that enables seamless integration with multiple AI models and providers:

1. **Provider Abstraction**:
   - Uniform interface to interact with models from different providers (Ollama, Hugging Face, OpenAI)
   - Standardized error handling and response formatting regardless of model source
   - Simplified switching between different model providers without application code changes

2. **Dynamic Model Selection**:
   - Runtime selection of appropriate models based on task requirements
   - Automatic quantization level selection based on available hardware
   - Fallback mechanisms when primary models are unavailable

3. **Multi-Modal Support**:
   - Single API for handling text, image, and audio embedding generation
   - Consistent interface for different model types (embedding, completion, classification)
   - Support for cross-modal operations where applicable

4. **Performance Optimization**:
   - Intelligent batching of requests to maximize throughput
   - Efficient resource allocation based on model requirements
   - Parallelization of operations when beneficial

5. **Integration Advantages**:
   - Enables easy addition of new model providers with minimal code changes
   - Facilitates A/B testing between different models for optimization
   - Provides monitoring and telemetry across all model interactions

This architecture allows OmniRAG to leverage the best models for each specific task while maintaining a clean separation of concerns in the codebase and providing flexibility for future model improvements.

### LightRAG Methodology

OmniRAG implements the LightRAG methodology, which focuses on enhancing retrieval efficiency and reducing token usage:

1. **Multi-tier Caching**:
   - Embeddings are cached in Redis to avoid recalculation
   - Query results are cached using a semantic fingerprinting mechanism
   - Document chunks are cached with intelligent expiration policies

2. **Token Optimization**:
   - Implements progressive retrieval that increases context only when necessary
   - Uses a chunk-merging algorithm to consolidate related information
   - Performs contextual compression to reduce redundancies before LLM processing

3. **Ranking Refinement**:
   - Applies a re-ranking step using cross-attention mechanisms
   - Filters out lower-quality chunks based on content quality heuristics
   - Implements dynamic context window management to maximize relevant information

4. **Adaptive Retrieval**:
   - Automatically adjusts retrieval parameters based on query complexity
   - Selects the optimal retriever combination for each query type
   - Scales context window based on semantic distance between chunks

### Hybrid Vector Search Pipeline

OmniRAG employs a sophisticated hybrid search pipeline that combines multiple retrieval techniques:

1. **Dense-Sparse Fusion**:
   - **Dense Vectors**: Semantic embeddings using state-of-the-art models
   - **Sparse Vectors**: BM25-based lexical search using Qdrant/bm25 model
   - **Fusion Algorithm**: Reciprocal rank fusion to combine results from both approaches

2. **Vector Database Integration**:
   - Qdrant for efficient similarity search across millions of vectors
   - Support for both exact and approximate nearest neighbor search
   - Vector filtering based on metadata attributes

3. **Knowledge Graph Augmentation**:
   - Neo4j vector indexes for entity and relationship embeddings
   - Graph traversal to discover related concepts not found in direct retrieval
   - Entity-centric search that understands relationships between concepts

4. **Multi-Stage Retrieval Process**:
   ```
   Query → Query Analysis → Hybrid Search (Dense + Sparse) → 
   Knowledge Graph Enrichment → Re-ranking → Response Generation
   ```

5. **Implementation Details**:
   - Embedding generation using both Hugging Face models and custom embeddings
   - Automated caching of embeddings with intelligent key generation
   - Parallel execution of dense and sparse vector creation
   - Score normalization and weighted combination of results

### Entity and Relationship Extraction

OmniRAG implements a sophisticated approach to entity extraction:

1. **Cross-Chunk Entity Registry**: Maintains entity consistency across document chunks
2. **Two-Pass Relationship Extraction**: Extracts and refines relationships across chunks
3. **Metadata as Entities**: Converts document metadata into searchable entities
4. **Neo4j Knowledge Graph**: Stores relationships for complex query answering

### Search Orchestration

The search process follows a multi-step workflow:

1. **Query Analysis**: Breaks down complex queries into searchable components
2. **Preliminary Search**: Performs initial retrieval to assess search strength
3. **Optimized Search**: Adapts search strategy based on preliminary results
4. **Result Verification**: Evaluates adequacy of search results
5. **Follow-up Search**: Performs additional searches if needed
6. **Result Ranking**: Deduplicates and ranks final results

## Configuration

Configuration parameters can be adjusted in the `.env` file. Key settings include:
- Vector database connection details (Qdrant)
- Neo4j connection parameters
- Redis cache configuration
- Minio object storage settings
- Embedding model selection
- Ollama model parameters
- API security settings
- Agent workflow configurations
- LightRAG optimization parameters
- Hybrid search weighting coefficients
- Batch processing limits
- Logging and telemetry options

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

