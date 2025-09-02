"""Main online inference pipeline orchestrator."""

import asyncio
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from ..core.config import settings
from ..core.exceptions import InferenceException
from ..core.logging_config import get_logger, log_performance
from .hyde_ensemble import hyde_ensemble
from .hybrid_search import hybrid_search
from .reranker.reranker_factory import RerankerFactory
from .reranker.base_reranker import RerankCandidate
from .context_packaging import context_packager
from ..knowledge_base.adjustment_engine import kb_adjustment_engine, FeedbackType


class InferencePipelineStage(Enum):
    """Stages of the inference pipeline."""
    INITIALIZATION = "initialization"
    HYDE_GENERATION = "hyde_generation"
    HYBRID_SEARCH = "hybrid_search"
    RERANKING = "reranking"
    CONTEXT_PACKAGING = "context_packaging"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class InferenceRequest:
    """Container for inference request data."""
    request_id: str
    query: str
    top_k: int
    filters: Optional[Dict[str, Any]]
    include_parent_chunks: bool
    context_window_tokens: int
    user_id: Optional[str]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "query": self.query,
            "top_k": self.top_k,
            "filters": self.filters,
            "include_parent_chunks": self.include_parent_chunks,
            "context_window_tokens": self.context_window_tokens,
            "user_id": self.user_id,
            "metadata": self.metadata
        }


@dataclass
class InferenceResponse:
    """Container for inference response data."""
    request_id: str
    query: str
    contexts: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    performance_metrics: Dict[str, float]
    pipeline_stage: InferencePipelineStage
    success: bool
    error: Optional[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "query": self.query,
            "contexts": self.contexts,
            "metadata": self.metadata,
            "performance_metrics": self.performance_metrics,
            "pipeline_stage": self.pipeline_stage.value,
            "success": self.success,
            "error": self.error
        }


class InferencePipeline:
    """Main online inference pipeline orchestrator."""
    
    def __init__(self):
        self.logger = get_logger(__name__, "inference_pipeline")
        self.reranker = None
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize the inference pipeline components."""
        if self._initialized:
            return
        
        try:
            start_time = time.time()
            
            # Initialize reranker
            self.reranker = RerankerFactory.create_reranker()
            
            # Warm up components
            self._warmup_components()
            
            self._initialized = True
            
            init_time = (time.time() - start_time) * 1000
            
            self.logger.info(
                f"Inference pipeline initialized successfully",
                extra={"init_time_ms": init_time}
            )
            
        except Exception as e:
            raise InferenceException(
                f"Failed to initialize inference pipeline: {str(e)}",
                component="inference_pipeline",
                error_code="PIPELINE_INIT_FAILED"
            )
    
    def _warmup_components(self) -> None:
        """Warm up pipeline components to reduce first-request latency."""
        try:
            # Warm up HyDE ensemble with a test query
            test_query = "What is artificial intelligence?"
            hyde_ensemble.generate_hypothetical_documents(test_query)
            
            # Warm up reranker if it has initialize method
            if hasattr(self.reranker, 'initialize'):
                self.reranker.initialize()
            
            self.logger.info("Pipeline components warmed up successfully")
            
        except Exception as e:
            self.logger.warning(f"Component warmup failed: {str(e)}")
    
    def process_query(
        self,
        query: str,
        top_k: int = None,
        filters: Optional[Dict[str, Any]] = None,
        include_parent_chunks: bool = True,
        context_window_tokens: int = None,
        user_id: Optional[str] = None,
        request_metadata: Optional[Dict[str, Any]] = None
    ) -> InferenceResponse:
        """Process a query through the complete inference pipeline."""
        
        # Create request
        request = InferenceRequest(
            request_id=str(uuid.uuid4()),
            query=query.strip(),
            top_k=top_k or settings.rerank_top_k,
            filters=filters,
            include_parent_chunks=include_parent_chunks,
            context_window_tokens=context_window_tokens or 4000,
            user_id=user_id,
            metadata=request_metadata or {}
        )
        
        # Initialize if needed
        if not self._initialized:
            self.initialize()
        
        # Validate request
        self._validate_request(request)
        
        # Process through pipeline
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self._execute_pipeline(request))
    
    async def process_query_async(
        self,
        query: str,
        top_k: int = None,
        filters: Optional[Dict[str, Any]] = None,
        include_parent_chunks: bool = True,
        context_window_tokens: int = None,
        user_id: Optional[str] = None,
        request_metadata: Optional[Dict[str, Any]] = None
    ) -> InferenceResponse:
        """Asynchronously process a query through the inference pipeline."""
        # Create request
        request = InferenceRequest(
            request_id=str(uuid.uuid4()),
            query=query.strip(),
            top_k=top_k or settings.rerank_top_k,
            filters=filters,
            include_parent_chunks=include_parent_chunks,
            context_window_tokens=context_window_tokens or 4000,
            user_id=user_id,
            metadata=request_metadata or {}
        )
        
        # Initialize if needed
        if not self._initialized:
            self.initialize()
        
        # Validate request
        self._validate_request(request)
        
        # Process through pipeline asynchronously
        return await self._execute_pipeline(request)
    
    def _validate_request(self, request: InferenceRequest) -> None:
        """Validate inference request."""
        if not request.query or not request.query.strip():
            raise InferenceException(
                "Empty query provided",
                component="inference_pipeline",
                error_code="EMPTY_QUERY",
                details={"request_id": request.request_id}
            )
        
        if request.top_k <= 0 or request.top_k > 100:
            raise InferenceException(
                f"Invalid top_k value: {request.top_k}",
                component="inference_pipeline",
                error_code="INVALID_TOP_K",
                details={"request_id": request.request_id, "top_k": request.top_k}
            )
    
    async def _execute_pipeline(self, request: InferenceRequest) -> InferenceResponse:
        """Execute the complete inference pipeline."""
        start_time = time.time()
        stage_metrics = {}
        current_stage = InferencePipelineStage.INITIALIZATION
        
        try:
            self.logger.info(
                f"Starting inference pipeline for request {request.request_id}",
                extra={
                    "request_id": request.request_id,
                    "query": request.query[:100],
                    "top_k": request.top_k
                }
            )
            
            # Stage 1: HyDE Generation
            current_stage = InferencePipelineStage.HYDE_GENERATION
            stage_start = time.time()
            
            hypothetical_docs = hyde_ensemble.generate_hypothetical_documents(request.query)
            
            stage_metrics["hyde_generation_ms"] = (time.time() - stage_start) * 1000
            stage_metrics["hypothetical_docs_count"] = len(hypothetical_docs)
            
            # Stage 2: Hybrid Search
            current_stage = InferencePipelineStage.HYBRID_SEARCH
            stage_start = time.time()
            
            search_results = await hybrid_search.search(
                query=request.query,
                hypothetical_docs=hypothetical_docs,
                top_k=request.top_k * 3,  # Get more candidates for reranking
                filters=request.filters
            )
            
            stage_metrics["hybrid_search_ms"] = (time.time() - stage_start) * 1000
            stage_metrics["search_candidates_count"] = len(search_results)
            
            # Stage 3: Reranking
            current_stage = InferencePipelineStage.RERANKING
            stage_start = time.time()
            
            # Convert search results to rerank candidates
            rerank_candidates = [
                RerankCandidate(
                    chunk_id=result.chunk_id,
                    content=result.content,
                    metadata=result.metadata,
                    original_score=result.combined_score
                )
                for result in search_results
            ]
            
            reranked_results = self.reranker.rerank(
                query=request.query,
                candidates=rerank_candidates,
                top_k=request.top_k
            )
            
            stage_metrics["reranking_ms"] = (time.time() - stage_start) * 1000
            stage_metrics["reranked_results_count"] = len(reranked_results)
            
            # Stage 4: Context Packaging
            current_stage = InferencePipelineStage.CONTEXT_PACKAGING
            stage_start = time.time()
            
            packaged_context = context_packager.package_context(
                reranked_results=reranked_results,
                query=request.query,
                include_parent_chunks=request.include_parent_chunks,
                context_window_tokens=request.context_window_tokens
            )
            
            stage_metrics["context_packaging_ms"] = (time.time() - stage_start) * 1000
            stage_metrics["final_contexts_count"] = len(packaged_context.contexts)
            stage_metrics["total_context_tokens"] = packaged_context.total_tokens
            
            # Calculate total time
            total_time = (time.time() - start_time) * 1000
            stage_metrics["total_pipeline_ms"] = total_time
            
            # Create successful response
            current_stage = InferencePipelineStage.COMPLETED
            
            response = InferenceResponse(
                request_id=request.request_id,
                query=request.query,
                contexts=packaged_context.contexts,
                metadata={
                    **packaged_context.metadata,
                    **request.metadata,
                    "pipeline_version": "1.0",
                    "components_used": {
                        "hyde_ensemble": hyde_ensemble.get_ensemble_info(),
                        "hybrid_search": hybrid_search.get_search_config(),
                        "reranker": self.reranker.get_model_info(),
                        "context_packager": {
                            "max_tokens": context_packager.max_context_tokens
                        }
                    }
                },
                performance_metrics=stage_metrics,
                pipeline_stage=current_stage,
                success=True,
                error=None
            )
            
            # Log performance
            log_performance(
                self.logger,
                "inference_pipeline",
                total_time,
                metadata={
                    "request_id": request.request_id,
                    "query_length": len(request.query),
                    "final_contexts": len(packaged_context.contexts),
                    **stage_metrics
                }
            )
            
            self.logger.info(
                f"Inference pipeline completed successfully for {request.request_id}",
                extra={
                    "request_id": request.request_id,
                    "total_time_ms": total_time,
                    "contexts_returned": len(packaged_context.contexts)
                }
            )
            
            return response
            
        except Exception as e:
            # Handle pipeline failure
            error_time = (time.time() - start_time) * 1000
            stage_metrics["total_pipeline_ms"] = error_time
            stage_metrics["error_stage"] = current_stage.value
            
            error_msg = str(e)
            
            self.logger.error(
                f"Inference pipeline failed at stage {current_stage.value}: {error_msg}",
                extra={
                    "request_id": request.request_id,
                    "error_stage": current_stage.value,
                    "error_time_ms": error_time
                },
                exc_info=True
            )
            
            return InferenceResponse(
                request_id=request.request_id,
                query=request.query,
                contexts=[],
                metadata=request.metadata,
                performance_metrics=stage_metrics,
                pipeline_stage=InferencePipelineStage.FAILED,
                success=False,
                error=error_msg
            )
    
    def record_feedback(
        self,
        request_id: str,
        chunk_id: str,
        feedback_type: str,
        relevance_score: float,
        query: str,
        user_rating: Optional[float] = None,
        comment: Optional[str] = None
    ) -> str:
        """Record user feedback for pipeline results."""
        try:
            # Convert string feedback type to enum
            feedback_enum = FeedbackType(feedback_type.lower())
            
            event_id = kb_adjustment_engine.record_feedback(
                chunk_id=chunk_id,
                query=query,
                feedback_type=feedback_enum,
                relevance_score=relevance_score,
                user_rating=user_rating,
                comment=comment,
                context_data={"request_id": request_id}
            )
            
            self.logger.info(
                f"Recorded feedback for request {request_id}",
                extra={
                    "request_id": request_id,
                    "chunk_id": chunk_id,
                    "feedback_type": feedback_type,
                    "event_id": event_id
                }
            )
            
            return event_id
            
        except Exception as e:
            raise InferenceException(
                f"Failed to record feedback: {str(e)}",
                component="inference_pipeline",
                error_code="FEEDBACK_RECORDING_FAILED",
                details={
                    "request_id": request_id,
                    "chunk_id": chunk_id
                }
            )
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics and health information."""
        try:
            kb_stats = kb_adjustment_engine.get_analytics_summary()
            
            return {
                "pipeline_initialized": self._initialized,
                "components": {
                    "hyde_ensemble": hyde_ensemble.get_ensemble_info(),
                    "hybrid_search": hybrid_search.get_search_config(),
                    "reranker": self.reranker.get_model_info() if self.reranker else None,
                    "context_packager": {
                        "max_context_tokens": context_packager.max_context_tokens
                    }
                },
                "knowledge_base": kb_stats,
                "version": "1.0"
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get pipeline stats: {str(e)}")
            return {"error": str(e)}
    
    def health_check(self) -> Dict[str, Any]:
        """Perform pipeline health check."""
        health_status = {
            "healthy": True,
            "components": {},
            "timestamp": time.time()
        }
        
        try:
            # Check if initialized
            if not self._initialized:
                health_status["healthy"] = False
                health_status["components"]["initialization"] = "not_initialized"
                return health_status
            
            # Check reranker
            if self.reranker:
                health_status["components"]["reranker"] = "healthy"
            else:
                health_status["healthy"] = False
                health_status["components"]["reranker"] = "not_available"
            
            # Test a simple query (very lightweight)
            try:
                # Use async version for health check to avoid blocking
                loop = asyncio.get_event_loop()
                test_response = loop.run_until_complete(
                    self.process_query_async("test query", top_k=1)
                )
                if test_response.success:
                    health_status["components"]["pipeline"] = "healthy"
                else:
                    health_status["healthy"] = False
                    health_status["components"]["pipeline"] = "test_query_failed"
            except Exception:
                health_status["healthy"] = False
                health_status["components"]["pipeline"] = "test_query_error"
            
            return health_status
            
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "timestamp": time.time()
            }


# Global inference pipeline instance
inference_pipeline = InferencePipeline()