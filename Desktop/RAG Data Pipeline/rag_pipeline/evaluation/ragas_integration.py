"""RAGAs integration for RAG triad metrics evaluation."""

import asyncio
import time
import uuid
from typing import Any, Dict, List, Optional
from datasets import Dataset
import pandas as pd

try:
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

from ..core.config import settings
from ..core.exceptions import EvaluationException
from ..core.logging_config import get_logger, log_performance


class RAGASEvaluator:
    """RAGAs integration for comprehensive RAG evaluation."""
    
    def __init__(self):
        self.logger = get_logger(__name__, "ragas_evaluator")
        self.metrics = []
        self.evaluation_history: List[Dict[str, Any]] = []
        
        if not RAGAS_AVAILABLE:
            self.logger.warning("RAGAs not available. Install with: pip install ragas")
            return
        
        # Initialize RAGAs metrics (RAG triad focus)
        self.metrics = [
            faithfulness,      # How factually accurate is the answer to retrieved context
            answer_relevancy,  # How relevant is the answer to the query
            context_precision, # How relevant is retrieved context to the query
        ]
        
        # Add context recall if ground truth is available
        self.metrics_with_ground_truth = self.metrics + [context_recall]
    
    async def evaluate_response(
        self,
        query: str,
        contexts: List[str],
        generated_answer: str,
        ground_truth: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Evaluate a single RAG response using RAGAs metrics."""
        if not RAGAS_AVAILABLE:
            raise EvaluationException(
                "RAGAs not available for evaluation",
                component="ragas_evaluator",
                error_code="RAGAS_NOT_AVAILABLE"
            )
        
        try:
            start_time = time.time()
            evaluation_id = str(uuid.uuid4())
            
            self.logger.info(
                f"Starting RAGAs evaluation {evaluation_id}",
                extra={
                    "evaluation_id": evaluation_id,
                    "query_length": len(query),
                    "contexts_count": len(contexts),
                    "has_ground_truth": ground_truth is not None
                }
            )
            
            # Prepare dataset for RAGAs
            data = {
                "question": [query],
                "answer": [generated_answer],
                "contexts": [contexts],
            }
            
            # Add ground truth if available
            metrics_to_use = self.metrics
            if ground_truth:
                data["ground_truths"] = [ground_truth]
                metrics_to_use = self.metrics_with_ground_truth
            
            dataset = Dataset.from_dict(data)
            
            # Run evaluation
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                evaluate, 
                dataset, 
                metrics_to_use
            )
            
            # Extract metrics
            metrics = {}
            for metric_name, value in result.items():
                if hasattr(value, 'iloc'):  # Handle pandas Series
                    metrics[metric_name] = float(value.iloc[0]) if len(value) > 0 else 0.0
                else:
                    metrics[metric_name] = float(value)
            
            evaluation_time = (time.time() - start_time) * 1000
            
            # Store evaluation result
            evaluation_result = {
                "evaluation_id": evaluation_id,
                "query": query,
                "contexts_count": len(contexts),
                "generated_answer": generated_answer[:100] + "..." if len(generated_answer) > 100 else generated_answer,
                "has_ground_truth": ground_truth is not None,
                "metrics": metrics,
                "metadata": metadata or {},
                "evaluation_time_ms": evaluation_time,
                "timestamp": time.time()
            }
            
            self.evaluation_history.append(evaluation_result)
            
            log_performance(
                self.logger,
                "ragas_evaluation",
                evaluation_time,
                metadata={
                    "evaluation_id": evaluation_id,
                    "metrics": metrics,
                    "contexts_count": len(contexts)
                }
            )
            
            self.logger.info(
                f"RAGAs evaluation completed {evaluation_id}",
                extra={
                    "evaluation_id": evaluation_id,
                    "metrics": metrics,
                    "evaluation_time_ms": evaluation_time
                }
            )
            
            return {
                "evaluation_id": evaluation_id,
                "metrics": metrics,
                "evaluation_time_ms": evaluation_time
            }
            
        except Exception as e:
            raise EvaluationException(
                f"RAGAs evaluation failed: {str(e)}",
                component="ragas_evaluator",
                error_code="RAGAS_EVALUATION_FAILED",
                details={
                    "query": query[:100],
                    "contexts_count": len(contexts)
                }
            )
    
    async def evaluate_batch(
        self,
        queries: List[str],
        contexts_list: List[List[str]],
        generated_answers: List[str],
        ground_truths: Optional[List[str]] = None,
        metadata_list: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Evaluate a batch of RAG responses."""
        if not RAGAS_AVAILABLE:
            raise EvaluationException(
                "RAGAs not available for evaluation",
                component="ragas_evaluator",
                error_code="RAGAS_NOT_AVAILABLE"
            )
        
        try:
            start_time = time.time()
            batch_id = str(uuid.uuid4())
            batch_size = len(queries)
            
            self.logger.info(
                f"Starting batch RAGAs evaluation {batch_id}",
                extra={
                    "batch_id": batch_id,
                    "batch_size": batch_size,
                    "has_ground_truths": ground_truths is not None
                }
            )
            
            # Prepare dataset
            data = {
                "question": queries,
                "answer": generated_answers,
                "contexts": contexts_list,
            }
            
            metrics_to_use = self.metrics
            if ground_truths:
                data["ground_truths"] = ground_truths
                metrics_to_use = self.metrics_with_ground_truth
            
            dataset = Dataset.from_dict(data)
            
            # Run batch evaluation
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                evaluate,
                dataset,
                metrics_to_use
            )
            
            # Process results
            batch_metrics = {}
            individual_results = []
            
            for metric_name, values in result.items():
                if hasattr(values, 'tolist'):
                    metric_values = values.tolist()
                else:
                    metric_values = [values] if not isinstance(values, list) else values
                
                batch_metrics[f"{metric_name}_mean"] = sum(metric_values) / len(metric_values)
                batch_metrics[f"{metric_name}_min"] = min(metric_values)
                batch_metrics[f"{metric_name}_max"] = max(metric_values)
                
                # Store individual results
                for i, value in enumerate(metric_values):
                    if i >= len(individual_results):
                        individual_results.append({})
                    individual_results[i][metric_name] = value
            
            evaluation_time = (time.time() - start_time) * 1000
            
            log_performance(
                self.logger,
                "ragas_batch_evaluation", 
                evaluation_time,
                metadata={
                    "batch_id": batch_id,
                    "batch_size": batch_size,
                    "batch_metrics": batch_metrics
                }
            )
            
            return {
                "batch_id": batch_id,
                "batch_size": batch_size,
                "batch_metrics": batch_metrics,
                "individual_results": individual_results,
                "evaluation_time_ms": evaluation_time
            }
            
        except Exception as e:
            raise EvaluationException(
                f"Batch RAGAs evaluation failed: {str(e)}",
                component="ragas_evaluator",
                error_code="RAGAS_BATCH_EVALUATION_FAILED",
                details={"batch_size": len(queries)}
            )
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all evaluation metrics."""
        if not self.evaluation_history:
            return {
                "total_evaluations": 0,
                "metrics_summary": {}
            }
        
        # Aggregate metrics
        all_metrics = {}
        for evaluation in self.evaluation_history:
            for metric_name, value in evaluation["metrics"].items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(value)
        
        # Calculate summary statistics
        summary = {}
        for metric_name, values in all_metrics.items():
            summary[metric_name] = {
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "count": len(values)
            }
        
        return {
            "total_evaluations": len(self.evaluation_history),
            "metrics_summary": summary,
            "recent_evaluations": self.evaluation_history[-10:]  # Last 10 evaluations
        }
    
    def get_evaluation_by_id(self, evaluation_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific evaluation by ID."""
        for evaluation in self.evaluation_history:
            if evaluation["evaluation_id"] == evaluation_id:
                return evaluation
        return None
    
    def clear_history(self) -> None:
        """Clear evaluation history."""
        self.evaluation_history.clear()
        self.logger.info("RAGAs evaluation history cleared")


# Global RAGAs evaluator instance
ragas_evaluator = RAGASEvaluator()