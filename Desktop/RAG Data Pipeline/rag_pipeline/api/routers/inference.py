"""Inference API endpoints."""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

from ...inference.pipeline import inference_pipeline
from ...knowledge_base.adjustment_engine import kb_adjustment_engine, FeedbackType
from ...core.logging_config import get_logger


router = APIRouter(prefix="/inference", tags=["inference"])
logger = get_logger(__name__, "inference_api")


class QueryRequest(BaseModel):
    """Request model for query processing."""
    query: str = Field(..., description="The query to process", min_length=1, max_length=1000)
    top_k: int = Field(default=5, description="Number of results to return", ge=1, le=20)
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Optional filters for search")
    include_parent_chunks: bool = Field(default=True, description="Whether to include parent chunks")
    context_window_tokens: int = Field(default=4000, description="Context window size in tokens", ge=500, le=8000)
    user_id: Optional[str] = Field(default=None, description="Optional user identifier")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional request metadata")


class QueryResponse(BaseModel):
    """Response model for query processing."""
    request_id: str
    query: str
    contexts: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    performance_metrics: Dict[str, float]
    pipeline_stage: str
    success: bool
    error: Optional[str]


class FeedbackRequest(BaseModel):
    """Request model for feedback submission."""
    request_id: str = Field(..., description="The request ID to provide feedback for")
    chunk_id: str = Field(..., description="The chunk ID being rated")
    feedback_type: str = Field(..., description="Type of feedback", regex="^(positive|negative|irrelevant|outdated|incorrect)$")
    relevance_score: float = Field(..., description="Relevance score 0-1", ge=0.0, le=1.0)
    user_rating: Optional[float] = Field(default=None, description="User rating 1-5", ge=1.0, le=5.0)
    comment: Optional[str] = Field(default=None, description="Optional feedback comment", max_length=500)


class FeedbackResponse(BaseModel):
    """Response model for feedback submission."""
    event_id: str
    success: bool
    message: str


@router.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest, http_request: Request):
    """Process a query through the RAG inference pipeline."""
    try:
        request_id = getattr(http_request.state, 'request_id', 'unknown')
        
        logger.info(
            f"Processing query request",
            extra={
                "request_id": request_id,
                "query_length": len(request.query),
                "top_k": request.top_k,
                "user_id": request.user_id
            }
        )
        
        # Process query through pipeline
        response = await inference_pipeline.process_query_async(
            query=request.query,
            top_k=request.top_k,
            filters=request.filters,
            include_parent_chunks=request.include_parent_chunks,
            context_window_tokens=request.context_window_tokens,
            user_id=request.user_id,
            request_metadata=request.metadata or {}
        )
        
        logger.info(
            f"Query processed successfully",
            extra={
                "request_id": response.request_id,
                "contexts_returned": len(response.contexts),
                "total_time_ms": response.performance_metrics.get("total_pipeline_ms", 0)
            }
        )
        
        return QueryResponse(**response.to_dict())
        
    except Exception as e:
        logger.error(f"Query processing failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query processing failed: {str(e)}"
        )


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest, http_request: Request):
    """Submit feedback for a query result."""
    try:
        request_id = getattr(http_request.state, 'request_id', 'unknown')
        
        logger.info(
            f"Submitting feedback",
            extra={
                "request_id": request_id,
                "chunk_id": request.chunk_id,
                "feedback_type": request.feedback_type
            }
        )
        
        # Record feedback
        event_id = inference_pipeline.record_feedback(
            request_id=request.request_id,
            chunk_id=request.chunk_id,
            feedback_type=request.feedback_type,
            relevance_score=request.relevance_score,
            query="",  # Would need to store original query for this
            user_rating=request.user_rating,
            comment=request.comment
        )
        
        logger.info(
            f"Feedback recorded successfully",
            extra={
                "event_id": event_id,
                "chunk_id": request.chunk_id
            }
        )
        
        return FeedbackResponse(
            event_id=event_id,
            success=True,
            message="Feedback recorded successfully"
        )
        
    except Exception as e:
        logger.error(f"Feedback submission failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Feedback submission failed: {str(e)}"
        )


@router.get("/stats")
async def get_pipeline_stats():
    """Get pipeline statistics and health information."""
    try:
        stats = inference_pipeline.get_pipeline_stats()
        return {"success": True, "stats": stats}
        
    except Exception as e:
        logger.error(f"Failed to get pipeline stats: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get stats: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """Perform pipeline health check."""
    try:
        health_status = inference_pipeline.health_check()
        status_code = status.HTTP_200_OK if health_status["healthy"] else status.HTTP_503_SERVICE_UNAVAILABLE
        
        return JSONResponse(
            status_code=status_code,
            content=health_status
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "healthy": False,
                "error": str(e),
                "timestamp": time.time()
            }
        )