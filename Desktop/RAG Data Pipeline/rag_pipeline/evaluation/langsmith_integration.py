"""Langsmith integration for tracking and evaluation."""

import asyncio
import time
import uuid
from typing import Any, Dict, List, Optional
import json

try:
    from langsmith import Client
    from langsmith.evaluation import evaluate, LangChainStringEvaluator
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False

from ..core.config import settings
from ..core.exceptions import EvaluationException
from ..core.logging_config import get_logger, log_performance
from ..core.pii_detection import pii_detector


class LangSmithClient:
    """Langsmith integration for RAG pipeline tracking and evaluation."""
    
    def __init__(self):
        self.logger = get_logger(__name__, "langsmith_client")
        self.client = None
        self.project_name = settings.langsmith_project
        self.session_history: List[Dict[str, Any]] = []
        
        if not LANGSMITH_AVAILABLE:
            self.logger.warning("Langsmith not available. Install with: pip install langsmith")
            return
        
        if not settings.langsmith_api_key:
            self.logger.warning("Langsmith API key not provided. Tracking disabled.")
            return
        
        try:
            self.client = Client(api_key=settings.langsmith_api_key)
            self.logger.info(f"Langsmith client initialized for project: {self.project_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize Langsmith client: {str(e)}")
            self.client = None
    
    async def track_query_session(
        self,
        query: str,
        contexts: List[str],
        generated_answer: str,
        metadata: Optional[Dict[str, Any]] = None,
        performance_metrics: Optional[Dict[str, float]] = None
    ) -> str:
        """Track a query session in Langsmith."""
        if not self.client:
            return "tracking_disabled"
        
        try:
            session_id = str(uuid.uuid4())
            start_time = time.time()
            
            self.logger.info(
                f"Tracking query session {session_id}",
                extra={
                    "session_id": session_id,
                    "query_length": len(query),
                    "contexts_count": len(contexts)
                }
            )
            
            # Create session data
            session_data = {
                "session_id": session_id,
                "query": query,
                "contexts": contexts,
                "generated_answer": generated_answer,
                "metadata": metadata or {},
                "performance_metrics": performance_metrics or {},
                "timestamp": time.time()
            }
            
            # Track with Langsmith
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._track_session_sync,
                session_data
            )
            
            # Store locally
            self.session_history.append(session_data)
            
            tracking_time = (time.time() - start_time) * 1000
            
            log_performance(
                self.logger,
                "langsmith_tracking",
                tracking_time,
                metadata={
                    "session_id": session_id,
                    "contexts_count": len(contexts)
                }
            )
            
            return session_id
            
        except Exception as e:
            self.logger.error(f"Failed to track session: {str(e)}")
            return "tracking_failed"
    
    def _track_session_sync(self, session_data: Dict[str, Any]) -> None:
        """Synchronous session tracking for executor with PII sanitization."""
        try:
            # Sanitize all data before sending to external service
            sanitized_data = pii_detector.sanitize_for_external_service(session_data, "langsmith")
            
            # Create a run in Langsmith with sanitized data
            self.client.create_run(
                name="rag_query",
                run_type="chain",
                project_name=self.project_name,
                inputs={
                    "query": sanitized_data["query"],
                    "contexts_count": len(session_data["contexts"]),
                    "has_pii": pii_detector.has_pii(str(session_data))
                },
                outputs={
                    "answer": sanitized_data["generated_answer"],
                    "contexts": sanitized_data.get("contexts", [])[:3]  # Limited and sanitized
                },
                extra={
                    "metadata": sanitized_data.get("metadata", {}),
                    "performance_metrics": sanitized_data.get("performance_metrics", {}),
                    "pii_summary": pii_detector.get_pii_summary(str(session_data))
                }
            )
            
        except Exception as e:
            self.logger.error(f"Langsmith tracking error: {str(e)}")
    
    async def evaluate_with_langsmith(
        self,
        query: str,
        contexts: List[str],
        generated_answer: str,
        ground_truth: Optional[str] = None,
        custom_evaluators: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Evaluate using Langsmith's evaluation framework."""
        if not self.client:
            raise EvaluationException(
                "Langsmith client not available",
                component="langsmith_client",
                error_code="LANGSMITH_NOT_AVAILABLE"
            )
        
        try:
            evaluation_id = str(uuid.uuid4())
            start_time = time.time()
            
            self.logger.info(
                f"Starting Langsmith evaluation {evaluation_id}",
                extra={
                    "evaluation_id": evaluation_id,
                    "has_ground_truth": ground_truth is not None
                }
            )
            
            # Prepare evaluation data
            example_data = {
                "inputs": {"query": query, "contexts": contexts},
                "outputs": {"answer": generated_answer},
                "reference_outputs": {"ground_truth": ground_truth} if ground_truth else None
            }
            
            # Define evaluators
            evaluators = []
            
            # Add default string evaluators if ground truth is available
            if ground_truth:
                evaluators.extend([
                    LangChainStringEvaluator("qa"),
                    LangChainStringEvaluator("context_qa", context="\n".join(contexts))
                ])
            
            # Add custom evaluators
            if custom_evaluators:
                for evaluator_name in custom_evaluators:
                    try:
                        evaluators.append(LangChainStringEvaluator(evaluator_name))
                    except Exception as e:
                        self.logger.warning(f"Failed to add evaluator {evaluator_name}: {e}")
            
            # Run evaluation
            if evaluators:
                loop = asyncio.get_event_loop()
                results = await loop.run_in_executor(
                    None,
                    self._evaluate_sync,
                    example_data,
                    evaluators
                )
            else:
                results = {"message": "No evaluators available"}
            
            evaluation_time = (time.time() - start_time) * 1000
            
            evaluation_result = {
                "evaluation_id": evaluation_id,
                "results": results,
                "evaluation_time_ms": evaluation_time
            }
            
            log_performance(
                self.logger,
                "langsmith_evaluation",
                evaluation_time,
                metadata={
                    "evaluation_id": evaluation_id,
                    "evaluators_count": len(evaluators)
                }
            )
            
            return evaluation_result
            
        except Exception as e:
            raise EvaluationException(
                f"Langsmith evaluation failed: {str(e)}",
                component="langsmith_client",
                error_code="LANGSMITH_EVALUATION_FAILED"
            )
    
    def _evaluate_sync(self, example_data: Dict[str, Any], evaluators: List) -> Dict[str, Any]:
        """Synchronous evaluation for executor."""
        try:
            results = {}
            
            for evaluator in evaluators:
                try:
                    result = evaluator.evaluate_strings(
                        prediction=example_data["outputs"]["answer"],
                        reference=example_data.get("reference_outputs", {}).get("ground_truth"),
                        input=example_data["inputs"]["query"]
                    )
                    results[f"evaluator_{evaluator.evaluation_name}"] = result
                except Exception as e:
                    results[f"evaluator_{evaluator.evaluation_name}_error"] = str(e)
            
            return results
            
        except Exception as e:
            return {"error": str(e)}
    
    async def create_dataset(
        self,
        dataset_name: str,
        examples: List[Dict[str, Any]],
        description: Optional[str] = None
    ) -> str:
        """Create a dataset in Langsmith."""
        if not self.client:
            raise EvaluationException(
                "Langsmith client not available",
                component="langsmith_client",
                error_code="LANGSMITH_NOT_AVAILABLE"
            )
        
        try:
            self.logger.info(
                f"Creating dataset {dataset_name} with {len(examples)} examples"
            )
            
            loop = asyncio.get_event_loop()
            dataset = await loop.run_in_executor(
                None,
                self._create_dataset_sync,
                dataset_name,
                examples,
                description
            )
            
            self.logger.info(f"Dataset created successfully: {dataset_name}")
            return dataset.id
            
        except Exception as e:
            raise EvaluationException(
                f"Failed to create dataset: {str(e)}",
                component="langsmith_client",
                error_code="DATASET_CREATION_FAILED"
            )
    
    def _create_dataset_sync(
        self,
        dataset_name: str,
        examples: List[Dict[str, Any]],
        description: Optional[str]
    ):
        """Synchronous dataset creation for executor."""
        return self.client.create_dataset(
            dataset_name=dataset_name,
            description=description or f"RAG evaluation dataset with {len(examples)} examples"
        )
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of Langsmith tracking metrics."""
        if not self.session_history:
            return {
                "total_sessions": 0,
                "tracking_enabled": self.client is not None,
                "project_name": self.project_name
            }
        
        # Calculate summary statistics
        total_sessions = len(self.session_history)
        avg_contexts = sum(len(session["contexts"]) for session in self.session_history) / total_sessions
        
        # Performance metrics summary
        perf_metrics = []
        for session in self.session_history:
            if "performance_metrics" in session:
                perf_metrics.append(session["performance_metrics"])
        
        avg_latency = 0
        if perf_metrics and "total_pipeline_ms" in perf_metrics[0]:
            avg_latency = sum(m.get("total_pipeline_ms", 0) for m in perf_metrics) / len(perf_metrics)
        
        return {
            "total_sessions": total_sessions,
            "tracking_enabled": self.client is not None,
            "project_name": self.project_name,
            "avg_contexts_per_query": avg_contexts,
            "avg_latency_ms": avg_latency,
            "recent_sessions": self.session_history[-5:]  # Last 5 sessions
        }
    
    def clear_history(self) -> None:
        """Clear session history."""
        self.session_history.clear()
        self.logger.info("Langsmith session history cleared")


# Global Langsmith client instance
langsmith_client = LangSmithClient()