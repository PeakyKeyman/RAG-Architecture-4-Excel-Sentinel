"""Evaluation API endpoints."""

import time
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

from ...evaluation.ragas_integration import ragas_evaluator
from ...evaluation.langsmith_integration import langsmith_client
from ...knowledge_base.adjustment_engine import kb_adjustment_engine
from ...core.logging_config import get_logger


router = APIRouter(prefix="/evaluation", tags=["evaluation"])
logger = get_logger(__name__, "evaluation_api")


class EvaluationRequest(BaseModel):
    """Request model for RAG evaluation."""
    query: str = Field(..., description="The query that was processed")
    contexts: List[str] = Field(..., description="Retrieved contexts")
    generated_answer: str = Field(..., description="Generated answer")
    ground_truth: Optional[str] = Field(default=None, description="Optional ground truth answer")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata")


class EvaluationResponse(BaseModel):
    """Response model for RAG evaluation."""
    evaluation_id: str
    metrics: Dict[str, float]
    success: bool
    evaluation_time_ms: float


class KnowledgeBaseAnalysisRequest(BaseModel):
    """Request model for knowledge base analysis."""
    min_retrievals: int = Field(default=5, description="Minimum retrievals for analysis", ge=1)
    confidence_threshold: float = Field(default=0.7, description="Confidence threshold for recommendations", ge=0.0, le=1.0)


class KnowledgeBaseAnalysisResponse(BaseModel):
    """Response model for knowledge base analysis."""
    recommendations: List[Dict[str, Any]]
    analytics_summary: Dict[str, Any]
    analysis_time_ms: float


class AdjustmentRequest(BaseModel):
    """Request model for knowledge base adjustments."""
    recommendations: List[Dict[str, Any]] = Field(..., description="Recommendations to apply")
    dry_run: bool = Field(default=True, description="Whether to perform a dry run")


class AdjustmentResponse(BaseModel):
    """Response model for knowledge base adjustments."""
    applied: int
    skipped: int
    errors: int
    actions: Dict[str, int]
    dry_run: bool


@router.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_rag_response(request: EvaluationRequest, http_request: Request):
    """Evaluate a RAG response using RAGAs metrics."""
    try:
        request_id = getattr(http_request.state, 'request_id', 'unknown')
        start_time = time.time()
        
        logger.info(
            f"Evaluating RAG response",
            extra={
                "request_id": request_id,
                "query_length": len(request.query),
                "contexts_count": len(request.contexts),
                "has_ground_truth": request.ground_truth is not None
            }
        )
        
        # Perform RAGAs evaluation
        evaluation_result = await ragas_evaluator.evaluate_response(
            query=request.query,
            contexts=request.contexts,
            generated_answer=request.generated_answer,
            ground_truth=request.ground_truth,
            metadata=request.metadata or {}
        )
        
        evaluation_time = (time.time() - start_time) * 1000
        
        logger.info(
            f"RAG evaluation completed",
            extra={
                "request_id": request_id,
                "evaluation_id": evaluation_result["evaluation_id"],
                "metrics": evaluation_result["metrics"],
                "evaluation_time_ms": evaluation_time
            }
        )
        
        return EvaluationResponse(
            evaluation_id=evaluation_result["evaluation_id"],
            metrics=evaluation_result["metrics"],
            success=True,
            evaluation_time_ms=evaluation_time
        )
        
    except Exception as e:
        logger.error(f"RAG evaluation failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG evaluation failed: {str(e)}"
        )


@router.post("/analyze-knowledge-base", response_model=KnowledgeBaseAnalysisResponse)
async def analyze_knowledge_base(request: KnowledgeBaseAnalysisRequest, http_request: Request):
    """Analyze knowledge base and generate adjustment recommendations."""
    try:
        request_id = getattr(http_request.state, 'request_id', 'unknown')
        start_time = time.time()
        
        logger.info(
            f"Analyzing knowledge base",
            extra={
                "request_id": request_id,
                "min_retrievals": request.min_retrievals,
                "confidence_threshold": request.confidence_threshold
            }
        )
        
        # Perform analysis
        recommendations = kb_adjustment_engine.analyze_knowledge_base(
            min_retrievals=request.min_retrievals
        )
        
        # Filter by confidence threshold
        filtered_recommendations = [
            rec for rec in recommendations 
            if rec.confidence >= request.confidence_threshold
        ]
        
        # Get analytics summary
        analytics_summary = kb_adjustment_engine.get_analytics_summary()
        
        analysis_time = (time.time() - start_time) * 1000
        
        logger.info(
            f"Knowledge base analysis completed",
            extra={
                "request_id": request_id,
                "total_recommendations": len(recommendations),
                "filtered_recommendations": len(filtered_recommendations),
                "analysis_time_ms": analysis_time
            }
        )
        
        return KnowledgeBaseAnalysisResponse(
            recommendations=[rec.to_dict() for rec in filtered_recommendations],
            analytics_summary=analytics_summary,
            analysis_time_ms=analysis_time
        )
        
    except Exception as e:
        logger.error(f"Knowledge base analysis failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Knowledge base analysis failed: {str(e)}"
        )


@router.post("/apply-adjustments", response_model=AdjustmentResponse)
async def apply_knowledge_base_adjustments(request: AdjustmentRequest, http_request: Request):
    """Apply knowledge base adjustments based on recommendations."""
    try:
        request_id = getattr(http_request.state, 'request_id', 'unknown')
        
        logger.info(
            f"Applying knowledge base adjustments",
            extra={
                "request_id": request_id,
                "recommendations_count": len(request.recommendations),
                "dry_run": request.dry_run
            }
        )
        
        # Convert request data to recommendation objects
        from ...knowledge_base.adjustment_engine import AdjustmentRecommendation, AdjustmentAction, ChunkAnalytics
        
        recommendations = []
        for rec_data in request.recommendations:
            # This would need proper deserialization logic
            recommendations.append(rec_data)  # Simplified for now
        
        # Apply adjustments
        results = kb_adjustment_engine.apply_adjustments(
            recommendations=recommendations,
            dry_run=request.dry_run
        )
        
        logger.info(
            f"Knowledge base adjustments completed",
            extra={
                "request_id": request_id,
                "applied": results["applied"],
                "skipped": results["skipped"],
                "errors": results["errors"],
                "dry_run": request.dry_run
            }
        )
        
        return AdjustmentResponse(
            applied=results["applied"],
            skipped=results["skipped"],
            errors=results["errors"],
            actions=results["actions"],
            dry_run=request.dry_run
        )
        
    except Exception as e:
        logger.error(f"Knowledge base adjustments failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Knowledge base adjustments failed: {str(e)}"
        )


@router.get("/metrics")
async def get_evaluation_metrics():
    """Get evaluation metrics and statistics."""
    try:
        # Get analytics from knowledge base
        analytics = kb_adjustment_engine.get_analytics_summary()
        
        # Get evaluation metrics from RAGAs
        ragas_metrics = await ragas_evaluator.get_metrics_summary()
        
        # Get Langsmith metrics
        langsmith_metrics = await langsmith_client.get_metrics_summary()
        
        return {
            "success": True,
            "knowledge_base_analytics": analytics,
            "ragas_metrics": ragas_metrics,
            "langsmith_metrics": langsmith_metrics
        }
        
    except Exception as e:
        logger.error(f"Failed to get evaluation metrics: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get evaluation metrics: {str(e)}"
        )