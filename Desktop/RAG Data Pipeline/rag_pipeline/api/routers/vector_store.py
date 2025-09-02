"""Vector store management API endpoints."""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

from ...vector_store.vertex_vector_store import vector_store
from ...chunking.hierarchical_chunker import HierarchicalChunker
from ...models.chunk import Document
from ...core.logging_config import get_logger


router = APIRouter(prefix="/vector-store", tags=["vector-store"])
logger = get_logger(__name__, "vector_store_api")


class DocumentRequest(BaseModel):
    """Request model for document ingestion."""
    documents: List[Dict[str, Any]] = Field(..., description="List of documents to ingest")
    chunk_config: Optional[Dict[str, Any]] = Field(default=None, description="Optional chunking configuration")


class DocumentResponse(BaseModel):
    """Response model for document ingestion."""
    success: bool
    message: str
    upserted_count: int
    failed_count: int
    processing_time_ms: float


class SearchRequest(BaseModel):
    """Request model for vector search."""
    query: str = Field(..., description="Search query", min_length=1, max_length=1000)
    top_k: int = Field(default=10, description="Number of results to return", ge=1, le=100)
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Optional search filters")


class SearchResponse(BaseModel):
    """Response model for vector search."""
    query: str
    results: List[Dict[str, Any]]
    total_results: int
    search_time_ms: float


@router.post("/upsert", response_model=DocumentResponse)
async def upsert_documents(request: DocumentRequest, http_request: Request):
    """Ingest and upsert documents into the vector store."""
    try:
        request_id = getattr(http_request.state, 'request_id', 'unknown')
        start_time = time.time()
        
        logger.info(
            f"Upserting {len(request.documents)} documents",
            extra={
                "request_id": request_id,
                "document_count": len(request.documents)
            }
        )
        
        # Initialize chunker
        chunk_config = request.chunk_config or {}
        chunker = HierarchicalChunker(
            child_chunk_size=chunk_config.get("child_chunk_size"),
            parent_chunk_size=chunk_config.get("parent_chunk_size"),
            chunk_overlap=chunk_config.get("chunk_overlap")
        )
        
        # Process documents
        all_chunks = []
        for doc_data in request.documents:
            document = Document(
                document_id=doc_data["document_id"],
                content=doc_data["content"],
                metadata=doc_data.get("metadata", {}),
                source_path=doc_data.get("source_path")
            )
            
            chunks = chunker.chunk_document(document)
            
            # Convert chunks to dictionary format for vector store
            for chunk in chunks:
                chunk_dict = {
                    "chunk_id": chunk.chunk_id,
                    "parent_id": chunk.parent_id,
                    "content": chunk.content,
                    "document_id": document.document_id,
                    "metadata": chunk.metadata
                }
                all_chunks.append(chunk_dict)
        
        # Upsert to vector store
        result = vector_store.upsert_chunks(all_chunks)
        
        processing_time = (time.time() - start_time) * 1000
        
        logger.info(
            f"Document upsert completed",
            extra={
                "request_id": request_id,
                "upserted_count": result["upserted_count"],
                "failed_count": result["failed_count"],
                "processing_time_ms": processing_time
            }
        )
        
        return DocumentResponse(
            success=result["failed_count"] == 0,
            message=f"Processed {len(request.documents)} documents, upserted {result['upserted_count']} chunks",
            upserted_count=result["upserted_count"],
            failed_count=result["failed_count"],
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Document upsert failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document upsert failed: {str(e)}"
        )


@router.post("/search", response_model=SearchResponse)
async def search_vectors(request: SearchRequest, http_request: Request):
    """Perform vector similarity search."""
    try:
        request_id = getattr(http_request.state, 'request_id', 'unknown')
        start_time = time.time()
        
        logger.info(
            f"Performing vector search",
            extra={
                "request_id": request_id,
                "query_length": len(request.query),
                "top_k": request.top_k
            }
        )
        
        # Perform search
        results = await vector_store.similarity_search_async(
            query=request.query,
            top_k=request.top_k,
            filters=request.filters
        )
        
        search_time = (time.time() - start_time) * 1000
        
        logger.info(
            f"Vector search completed",
            extra={
                "request_id": request_id,
                "results_count": len(results),
                "search_time_ms": search_time
            }
        )
        
        return SearchResponse(
            query=request.query,
            results=[result.to_dict() for result in results],
            total_results=len(results),
            search_time_ms=search_time
        )
        
    except Exception as e:
        logger.error(f"Vector search failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Vector search failed: {str(e)}"
        )


@router.get("/stats")
async def get_index_stats():
    """Get vector index statistics."""
    try:
        stats = vector_store.get_index_stats()
        return {"success": True, "stats": stats}
        
    except Exception as e:
        logger.error(f"Failed to get index stats: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get index stats: {str(e)}"
        )