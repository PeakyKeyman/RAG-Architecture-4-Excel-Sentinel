# RAG Data Pipeline

A production-ready Retrieval-Augmented Generation (RAG) pipeline built following the Aegis Fidelity Protocol. This system provides enterprise-grade document ingestion, hierarchical chunking, hybrid search, and intelligent reranking capabilities.

## Features

- **Hierarchical Chunking**: Parent-child chunk relationships with configurable sizes and overlap
- **Hybrid Search**: Vector similarity + text search with Reciprocal Rank Fusion (RRF)
- **Ensemble HyDE**: Multiple hypothesis generation for enhanced retrieval
- **Intelligent Reranking**: Cohere Rerank API integration for precision optimization
- **Vector Storage**: Google Vertex AI Vector Search for scalable similarity search
- **Context Packaging**: Smart context window optimization with knapsack algorithm
- **Knowledge Base Adjustment**: Continuous improvement through feedback loops
- **Comprehensive Evaluation**: RAGAs metrics integration with Langsmith tracking
- **Production API**: FastAPI with authentication, rate limiting, and monitoring
- **Thread-Safe Architecture**: Concurrent request handling with proper synchronization

## Quick Start

### Prerequisites

- Python 3.9+
- Google Cloud Project with Vertex AI enabled
- Cohere API key
- Gemini API key (for HyDE)
- Langsmith API key (for evaluation)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd rag-data-pipeline

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GEMINI_API_KEY="your-gemini-api-key"
export COHERE_API_KEY="your-cohere-api-key" 
export LANGSMITH_API_KEY="your-langsmith-api-key"
```

### Start the API Server

```bash
uvicorn rag_pipeline.api.main:app --host 0.0.0.0 --port 8000
```

### Basic Usage

```python
import requests

# Add documents
response = requests.post("http://localhost:8000/documents", json={
    "documents": [{
        "document_id": "doc_001",
        "content": "Machine learning is a subset of artificial intelligence...",
        "metadata": {"source": "ml_textbook.pdf", "category": "education"}
    }]
})

# Query the system
response = requests.post("http://localhost:8000/query", json={
    "query": "What is machine learning?",
    "top_k": 5
})

result = response.json()
print(f"Answer: {result['answer']}")
```

## Architecture

The pipeline follows a modular architecture with clear separation of concerns:

```
rag_pipeline/
├── core/                    # Configuration, logging, exceptions
├── models/                  # Data models and contracts
├── vector_store/           # Embedding models and vector operations
├── chunking/               # Hierarchical document chunking
├── inference/              # Search, reranking, and HyDE
├── knowledge_base/         # Feedback processing and adjustment
├── evaluation/             # RAGAs metrics and evaluation
├── api/                    # FastAPI routers and middleware
└── tests/                  # Comprehensive test suite
```

### Key Components

1. **Hierarchical Chunker**: Creates parent-child chunk relationships for better context preservation
2. **Embedding Model**: Thread-safe BAAI/bge-large-en-v1.5 with preprocessing
3. **Vector Store**: Vertex AI integration with distance-to-similarity conversion
4. **Hybrid Search**: Combines vector and text search with RRF fusion
5. **Ensemble HyDE**: Multiple Gemini models for hypothesis generation
6. **Reranker**: Cohere API integration with validation and error handling
7. **Context Packager**: Knapsack optimization for context window utilization
8. **Knowledge Base Engine**: Feedback-driven continuous improvement

## Configuration

The system uses a hierarchical configuration approach:

1. **Default values** in `config.yaml`
2. **Environment variables** (highest priority)
3. **Runtime overrides** via API

Key configuration sections:

```yaml
# Chunking parameters
chunking:
  child_chunk_size: 250      # Tokens per child chunk
  parent_chunk_size: 750     # Tokens per parent chunk  
  chunk_overlap: 0.15        # 15% overlap between chunks

# Search configuration
search:
  hybrid_fusion_weight: 0.6  # Vector vs text search balance
  rrf_constant: 60           # Reciprocal Rank Fusion parameter
  default_top_k: 20          # Default results per query

# Performance tuning
performance:
  request_timeout_seconds: 30
  max_concurrent_requests: 10
  vector_search_timeout_seconds: 10
```

## API Reference

### Core Endpoints

- **POST /query** - Process queries through the full RAG pipeline
- **POST /documents** - Add documents to the knowledge base
- **GET /documents/{id}** - Retrieve document information
- **POST /search/vector** - Vector similarity search only
- **POST /search/hybrid** - Hybrid search with optional reranking
- **POST /evaluate** - Run evaluation on datasets
- **GET /health** - System health check
- **GET /metrics** - Performance metrics

### Authentication

Optional API key authentication:

```bash
curl -H "X-API-Key: your-api-key" http://localhost:8000/query
```

### Rate Limiting

- Default: 60 requests per minute
- Burst limit: 10 concurrent requests
- Configurable per environment

## Evaluation

The system includes comprehensive evaluation capabilities using RAGAs metrics:

- **Faithfulness**: How well answers align with retrieved contexts
- **Answer Relevancy**: How relevant answers are to questions
- **Context Relevancy**: How relevant retrieved contexts are to questions

```python
# Run evaluation
response = requests.post("http://localhost:8000/evaluate", json={
    "evaluation_data": evaluation_dataset,
    "metrics": ["faithfulness", "answer_relevancy", "context_relevancy"]
})
```

Results are automatically tracked in Langsmith for analysis and comparison.

## Testing

Comprehensive test suite with 80%+ coverage:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=rag_pipeline --cov-report=html

# Run specific test categories
pytest tests/unit/          # Unit tests
pytest tests/integration/   # Integration tests  
pytest tests/e2e/          # End-to-end tests
```

## Monitoring and Observability

The system provides detailed logging and metrics:

- **Structured JSON logging** with request tracing
- **Performance metrics** via `/metrics` endpoint
- **Health checks** for all components
- **Request/response timing** and success rates
- **Component-level status** monitoring

## Performance

Designed for production workloads:

- **Latency**: 2-8 second query processing (15s max for beta)
- **Throughput**: 5-10 queries per second
- **Scalability**: 1k-100k document collections
- **Concurrency**: Thread-safe architecture with proper locking
- **Memory**: Efficient caching with LRU eviction

## Security

- **API key authentication** (optional)
- **Input validation** and sanitization
- **Rate limiting** to prevent abuse
- **Error handling** without information leakage
- **Secure logging** without sensitive data exposure

## Development

### Code Quality

- **Black** formatting
- **isort** import sorting
- **flake8** linting
- **mypy** type checking

```bash
# Format code
black rag_pipeline/
isort rag_pipeline/

# Check code quality
flake8 rag_pipeline/
mypy rag_pipeline/
```

### Contributing

1. Follow the Aegis Fidelity Protocol
2. Maintain 80%+ test coverage
3. Add comprehensive documentation
4. Follow existing architectural patterns
5. Ensure thread safety for concurrent operations

## Deployment

### Production Checklist

- [ ] Set all required environment variables
- [ ] Configure Google Cloud authentication
- [ ] Set up Vertex AI Vector Search index
- [ ] Configure rate limiting and authentication
- [ ] Set up monitoring and alerting
- [ ] Test with realistic data volumes
- [ ] Validate performance requirements

### Docker Deployment

```bash
# Build image
docker build -t rag-pipeline .

# Run container
docker run -p 8000:8000 \
  -e GOOGLE_CLOUD_PROJECT=your-project \
  -e GEMINI_API_KEY=your-key \
  -e COHERE_API_KEY=your-key \
  rag-pipeline
```

### Production Configuration

```yaml
# Production settings
api:
  workers: 4
  host: "0.0.0.0"
  port: 8000

performance:
  max_concurrent_requests: 20
  request_timeout_seconds: 15

auth:
  enabled: true
  api_key: ${API_KEY}
```

## Troubleshooting

### Common Issues

1. **Vertex AI Connection**: Ensure project ID and authentication are correct
2. **Memory Usage**: Monitor embedding model memory, consider CPU-only deployment
3. **Timeout Errors**: Check network connectivity to external APIs
4. **Rate Limiting**: Monitor API quotas for Cohere and Gemini
5. **Chunk Quality**: Tune chunking parameters for your document types

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
uvicorn rag_pipeline.api.main:app --reload
```

### Health Checks

```bash
# Check system health
curl http://localhost:8000/health

# Check metrics
curl http://localhost:8000/metrics
```

## License

[License information]

## Support

For questions and support:
- Review the API documentation
- Check usage examples
- File issues with detailed reproduction steps
- Follow the troubleshooting guide