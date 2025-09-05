# RAG Pipeline API Documentation

## Overview

The RAG Pipeline API provides endpoints for document ingestion, query processing, and evaluation. The API is built with FastAPI and follows RESTful principles.

## Base URL

```
http://localhost:8000
```

## Authentication

API key authentication is optional but recommended for production use.

```bash
# Set API key via header
curl -H "X-API-Key: your-api-key" http://localhost:8000/query
```

## Core Endpoints

### 1. Query Processing

**POST /query**

Process a query through the complete RAG pipeline.

**Request Body:**
```json
{
  "query": "What is machine learning?",
  "top_k": 5,
  "filters": {
    "document_type": "research_paper"
  },
  "include_debug": false
}
```

**Response:**
```json
{
  "query": "What is machine learning?",
  "answer": "Machine learning is a subset of artificial intelligence...",
  "sources": [
    {
      "chunk_id": "chunk_123",
      "content": "Machine learning algorithms...",
      "score": 0.92,
      "metadata": {
        "source": "ml_textbook.pdf",
        "page": 15
      }
    }
  ],
  "processing_time_ms": 1250,
  "debug_info": null
}
```

### 2. Document Management

**POST /documents**

Add documents to the knowledge base.

**Request Body:**
```json
{
  "documents": [
    {
      "document_id": "doc_001",
      "content": "Document content here...",
      "metadata": {
        "source": "document.pdf",
        "type": "research_paper",
        "author": "John Doe"
      }
    }
  ]
}
```

**Response:**
```json
{
  "processed_documents": 1,
  "chunk_count": 15,
  "processing_time_ms": 3500
}
```

**GET /documents/{document_id}**

Retrieve document information and chunks.

**Response:**
```json
{
  "document_id": "doc_001",
  "metadata": {
    "source": "document.pdf",
    "type": "research_paper"
  },
  "chunks": [
    {
      "chunk_id": "chunk_001",
      "content": "Chunk content...",
      "parent_id": null,
      "token_count": 245
    }
  ],
  "total_chunks": 15
}
```

### 3. Search Operations

**POST /search/vector**

Perform vector similarity search.

**Request Body:**
```json
{
  "query": "machine learning algorithms",
  "top_k": 10,
  "filters": {
    "document_type": "research_paper"
  }
}
```

**POST /search/hybrid**

Perform hybrid search (vector + text).

**Request Body:**
```json
{
  "query": "deep learning neural networks",
  "top_k": 10,
  "enable_hyde": true,
  "rerank": true,
  "filters": {}
}
```

### 4. Evaluation

**POST /evaluate**

Evaluate RAG performance on a dataset.

**Request Body:**
```json
{
  "evaluation_data": [
    {
      "question": "What is deep learning?",
      "ground_truth": "Deep learning is a subset of machine learning...",
      "contexts": ["Relevant context 1", "Relevant context 2"]
    }
  ],
  "metrics": ["faithfulness", "answer_relevancy", "context_relevancy"]
}
```

**Response:**
```json
{
  "evaluation_id": "eval_123",
  "metrics": {
    "faithfulness": 0.85,
    "answer_relevancy": 0.92,
    "context_relevancy": 0.78
  },
  "sample_count": 100,
  "evaluation_time_ms": 45000
}
```

### 5. System Status

**GET /health**

Check system health status.

**Response:**
```json
{
  "status": "healthy",
  "components": {
    "vector_store": "connected",
    "embedding_model": "loaded",
    "reranker": "available"
  },
  "uptime_seconds": 3600
}
```

**GET /metrics**

Get system performance metrics.

**Response:**
```json
{
  "requests_per_second": 2.5,
  "avg_response_time_ms": 850,
  "cache_hit_rate": 0.65,
  "total_documents": 1500,
  "total_chunks": 12000
}
```

## Error Responses

All endpoints return standardized error responses:

```json
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "Query parameter is required",
    "details": "The 'query' field must be a non-empty string"
  },
  "request_id": "req_abc123"
}
```

Common error codes:
- `INVALID_REQUEST` (400): Malformed request
- `AUTHENTICATION_FAILED` (401): Invalid API key
- `RATE_LIMIT_EXCEEDED` (429): Too many requests
- `INTERNAL_ERROR` (500): Server error

## Rate Limiting

The API implements rate limiting with the following defaults:
- 60 requests per minute per API key
- Burst limit of 10 concurrent requests

Rate limit headers are included in responses:
```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 55
X-RateLimit-Reset: 1609459200
```

## WebSocket Support

**WS /stream**

Real-time query processing with streaming responses.

```javascript
const ws = new WebSocket('ws://localhost:8000/stream');
ws.send(JSON.stringify({
  "query": "Tell me about machine learning",
  "stream": true
}));

ws.onmessage = function(event) {
  const data = JSON.parse(event.data);
  console.log(data.chunk); // Partial response
};
```