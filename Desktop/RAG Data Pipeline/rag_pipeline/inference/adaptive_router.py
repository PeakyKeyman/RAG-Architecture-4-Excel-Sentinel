"""
Adaptive RAG router for query complexity-based strategy selection.

Routes queries to optimal retrieval strategies based on complexity analysis:
- Simple: Direct vector search (3x faster)
- Medium: Hybrid search with reranking (current default)
- Complex: Full ensemble with multi-step reasoning
"""

import time
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from ..core.logging_config import get_logger, log_performance
from ..core.audit_logging import security_auditor
from .query_classifier import query_classifier, QueryComplexity
from .hybrid_search import hybrid_search
from .hyde_ensemble import hyde_ensemble
from .reranker.reranker_factory import RerankerFactory
from .temporal_scorer import temporal_scorer
from ..vector_store.vertex_vector_store import vector_store


logger = get_logger(__name__, "adaptive_router")


class RetrievalStrategy(Enum):
    """Available retrieval strategies."""
    DIRECT_VECTOR = "direct_vector"      # Simple queries - vector search only
    HYBRID_RERANK = "hybrid_rerank"      # Medium queries - hybrid + rerank  
    FULL_ENSEMBLE = "full_ensemble"      # Complex queries - HyDE + hybrid + rerank


@dataclass
class RoutingDecision:
    """Result of adaptive routing decision."""
    strategy: RetrievalStrategy
    complexity: QueryComplexity
    confidence: float
    reasoning: list
    estimated_time_ms: float
    features_enabled: Dict[str, bool] = field(default_factory=dict)


@dataclass
class RoutingMetrics:
    """Performance metrics for routing decisions."""
    total_queries: int = 0
    simple_queries: int = 0
    medium_queries: int = 0
    complex_queries: int = 0
    avg_classification_time_ms: float = 0.0
    avg_retrieval_time_ms: float = 0.0
    accuracy_feedback: Dict[str, int] = field(default_factory=lambda: {
        'correct': 0, 'incorrect': 0
    })


class AdaptiveRAGRouter:
    """
    Intelligent query router for adaptive RAG strategies.
    
    Features:
    - Sub-millisecond query classification
    - Performance-optimized routing
    - Graceful degradation
    - Feature flags for A/B testing
    - Comprehensive metrics tracking
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize adaptive router with configuration."""
        self.config = config or {}
        self.logger = get_logger(__name__, "adaptive_router")
        
        # Feature flags (can be toggled via config)
        self.features = {
            'adaptive_routing_enabled': self.config.get('adaptive_routing_enabled', True),
            'simple_query_optimization': self.config.get('simple_query_optimization', True),
            'complex_query_ensemble': self.config.get('complex_query_ensemble', True),
            'performance_monitoring': self.config.get('performance_monitoring', True),
            'fallback_to_default': self.config.get('fallback_to_default', True)
        }
        
        # Performance targets (configurable)
        self.performance_targets = {
            'simple_query_max_ms': self.config.get('simple_query_max_ms', 500),
            'medium_query_max_ms': self.config.get('medium_query_max_ms', 2000), 
            'complex_query_max_ms': self.config.get('complex_query_max_ms', 8000)
        }
        
        # Initialize metrics
        self.metrics = RoutingMetrics()
        
        # Initialize reranker (shared across strategies)
        try:
            self.reranker = RerankerFactory.create_reranker()
            self.reranker_available = True
        except Exception as e:
            self.logger.warning(f"Reranker initialization failed: {str(e)}")
            self.reranker_available = False
    
    def route_query(self, 
                   query: str,
                   user_id: str = None,
                   org_id: str = None,
                   filters: Dict[str, Any] = None) -> RoutingDecision:
        """
        Route query to optimal retrieval strategy.
        
        Args:
            query: User query string
            user_id: User identifier for audit logging
            org_id: Organization identifier
            filters: Additional search filters
            
        Returns:
            RoutingDecision with selected strategy and reasoning
        """
        start_time = time.time()
        
        try:
            # Check if adaptive routing is enabled
            if not self.features['adaptive_routing_enabled']:
                return self._create_default_decision()
            
            # Classify query complexity
            classification = query_classifier.classify(query)
            
            # Determine routing strategy
            if classification.complexity == QueryComplexity.SIMPLE and self.features['simple_query_optimization']:
                strategy = RetrievalStrategy.DIRECT_VECTOR
                estimated_time = 300  # ~300ms for simple queries
                
            elif classification.complexity == QueryComplexity.COMPLEX and self.features['complex_query_ensemble']:
                strategy = RetrievalStrategy.FULL_ENSEMBLE
                estimated_time = 5000  # ~5s for complex ensemble
                
            else:
                # Medium complexity or feature flags disabled
                strategy = RetrievalStrategy.HYBRID_RERANK
                estimated_time = 1500  # ~1.5s for hybrid + rerank
            
            # Create routing decision
            decision = RoutingDecision(
                strategy=strategy,
                complexity=classification.complexity,
                confidence=classification.confidence,
                reasoning=classification.reasoning + [f"Selected {strategy.value}"],
                estimated_time_ms=estimated_time,
                features_enabled=self.features.copy()
            )
            
            # Update metrics
            self._update_routing_metrics(classification.complexity)
            
            # Audit logging
            if user_id:
                security_auditor.log_data_access(
                    user_id=user_id,
                    resource_type="query_routing",
                    resource_id=strategy.value,
                    access_level="standard",
                    document_count=None
                )
            
            classification_time = (time.time() - start_time) * 1000
            
            if self.features['performance_monitoring']:
                self.logger.info(
                    f"Query routed to {strategy.value}",
                    extra={
                        "complexity": classification.complexity.value,
                        "confidence": classification.confidence,
                        "classification_time_ms": classification_time,
                        "estimated_time_ms": estimated_time,
                        "user_id": user_id,
                        "org_id": org_id
                    }
                )
            
            return decision
            
        except Exception as e:
            self.logger.error(f"Routing failed: {str(e)}", exc_info=True)
            
            # Graceful degradation - fall back to default strategy
            if self.features['fallback_to_default']:
                return self._create_default_decision()
            else:
                raise
    
    async def execute_strategy(self,
                              decision: RoutingDecision,
                              query: str,
                              top_k: int = 20,
                              filters: Dict[str, Any] = None,
                              user_id: str = None,
                              org_id: str = None,
                              request_id: str = None) -> Dict[str, Any]:
        """
        Execute the selected retrieval strategy.
        
        Args:
            decision: Routing decision from route_query
            query: Original query string
            top_k: Number of results to retrieve
            filters: Search filters
            user_id: User identifier
            org_id: Organization identifier
            request_id: Request ID for performance tracking
            
        Returns:
            Dictionary with retrieved results and metadata
        """
        start_time = time.time()
        
        # Performance monitoring
        retrieval_timer = PerformanceTimer(f"retrieval_{decision.strategy.value}")
        
        try:
            with retrieval_timer:
                if decision.strategy == RetrievalStrategy.DIRECT_VECTOR:
                    results = await self._execute_direct_vector(query, top_k, filters, org_id, user_id)
                    
                elif decision.strategy == RetrievalStrategy.HYBRID_RERANK:
                    results = await self._execute_hybrid_rerank(query, top_k, filters, org_id, user_id)
                    
                elif decision.strategy == RetrievalStrategy.FULL_ENSEMBLE:
                    results = await self._execute_full_ensemble(query, top_k, filters, org_id, user_id)
                    
                else:
                    raise ValueError(f"Unknown strategy: {decision.strategy}")
            
            execution_time = retrieval_timer.elapsed_ms
            
            # Record detailed performance metrics
            if request_id and self.features['performance_monitoring']:
                record_query_performance(
                    query_id=request_id,
                    strategy=decision.strategy.value,
                    complexity=decision.complexity.value,
                    execution_time_ms=execution_time,
                    retrieval_time_ms=execution_time,  # For this level, retrieval time = execution time
                    total_chunks_retrieved=results.get('total_retrieved', 0),
                    final_chunks_returned=len(results.get('chunks', [])),
                    success=True
                )
            
            # Add execution metadata
            results['routing_metadata'] = {
                'strategy': decision.strategy.value,
                'complexity': decision.complexity.value,
                'confidence': decision.confidence,
                'execution_time_ms': execution_time,
                'estimated_time_ms': decision.estimated_time_ms,
                'performance_ratio': execution_time / decision.estimated_time_ms if decision.estimated_time_ms > 0 else 1.0
            }
            
            # Update performance metrics
            self._update_execution_metrics(execution_time)
            
            # Check performance against targets
            target_time = self.performance_targets.get(
                f"{decision.complexity.value}_query_max_ms", 
                decision.estimated_time_ms
            )
            
            if execution_time > target_time:
                self.logger.warning(
                    f"Query execution exceeded target: {execution_time:.1f}ms > {target_time}ms",
                    extra={
                        "strategy": decision.strategy.value,
                        "complexity": decision.complexity.value,
                        "target_time_ms": target_time,
                        "actual_time_ms": execution_time
                    }
                )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Strategy execution failed: {str(e)}", exc_info=True)
            
            # Record failed performance
            if request_id and self.features['performance_monitoring']:
                record_query_performance(
                    query_id=request_id,
                    strategy=decision.strategy.value,
                    complexity=decision.complexity.value,
                    execution_time_ms=retrieval_timer.elapsed_ms,
                    total_chunks_retrieved=0,
                    final_chunks_returned=0,
                    success=False,
                    error=str(e)
                )
            
            # Fallback to hybrid rerank on execution failure
            if decision.strategy != RetrievalStrategy.HYBRID_RERANK:
                self.logger.info("Falling back to hybrid_rerank strategy")
                return await self._execute_hybrid_rerank(query, top_k, filters, org_id, user_id)
            else:
                raise
    
    async def _execute_direct_vector(self, query: str, top_k: int, filters: Dict, org_id: str, user_id: str) -> Dict:
        """Execute direct vector search strategy (fastest)."""
        results = await vector_store.similarity_search_async(
            query=query,
            top_k=top_k,
            filters=filters
        )
        
        # Apply temporal scoring
        result_dicts = [result.to_dict() for result in results]
        scored_results = temporal_scorer.score_results(query, result_dicts)
        
        return {
            'chunks': scored_results,
            'strategy': 'direct_vector',
            'total_retrieved': len(results),
            'temporal_scoring_applied': True
        }
    
    async def _execute_hybrid_rerank(self, query: str, top_k: int, filters: Dict, org_id: str, user_id: str) -> Dict:
        """Execute hybrid search + reranking strategy (default)."""
        # Hybrid search with larger initial retrieval
        hybrid_results = await hybrid_search.search_async(
            query=query,
            top_k=top_k * 3,  # Get more candidates for reranking
            filters=filters
        )
        
        # Rerank if available
        if self.reranker_available and len(hybrid_results) > top_k:
            candidates = [
                {"content": result.content, "metadata": result.metadata}
                for result in hybrid_results
            ]
            
            reranked_results = await self.reranker.rerank_async(
                query=query,
                candidates=candidates,
                top_k=top_k
            )
            
            final_results = reranked_results[:top_k]
        else:
            final_results = hybrid_results[:top_k]
        
        # Apply temporal scoring
        result_dicts = [result.to_dict() if hasattr(result, 'to_dict') else result for result in final_results]
        scored_results = temporal_scorer.score_results(query, result_dicts)
        
        return {
            'chunks': scored_results,
            'strategy': 'hybrid_rerank',
            'total_retrieved': len(hybrid_results),
            'reranked': self.reranker_available,
            'final_count': len(scored_results),
            'temporal_scoring_applied': True
        }
    
    async def _execute_full_ensemble(self, query: str, top_k: int, filters: Dict, org_id: str, user_id: str) -> Dict:
        """Execute full ensemble strategy with HyDE (most comprehensive)."""
        # Generate hypothetical documents
        hypothetical_docs = hyde_ensemble.generate_hypothetical_documents(query)
        
        # Search with original query + hypothetical documents
        all_queries = [query] + hypothetical_docs
        
        all_results = []
        for q in all_queries:
            results = await hybrid_search.search_async(
                query=q,
                top_k=top_k,
                filters=filters
            )
            all_results.extend(results)
        
        # Deduplicate and rerank
        unique_results = self._deduplicate_results(all_results)
        
        if self.reranker_available and len(unique_results) > top_k:
            candidates = [
                {"content": result.content, "metadata": result.metadata}
                for result in unique_results
            ]
            
            reranked_results = await self.reranker.rerank_async(
                query=query,
                candidates=candidates,
                top_k=top_k
            )
            
            final_results = reranked_results[:top_k]
        else:
            final_results = unique_results[:top_k]
        
        # Apply temporal scoring
        result_dicts = [result.to_dict() if hasattr(result, 'to_dict') else result for result in final_results]
        scored_results = temporal_scorer.score_results(query, result_dicts)
        
        return {
            'chunks': scored_results,
            'strategy': 'full_ensemble',
            'hyde_queries': len(hypothetical_docs),
            'total_retrieved': len(all_results),
            'unique_results': len(unique_results),
            'reranked': self.reranker_available,
            'final_count': len(scored_results),
            'temporal_scoring_applied': True
        }
    
    def _deduplicate_results(self, results: list) -> list:
        """Remove duplicate results based on chunk_id."""
        seen_ids = set()
        unique_results = []
        
        for result in results:
            chunk_id = getattr(result, 'chunk_id', str(hash(str(result))))
            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                unique_results.append(result)
        
        return unique_results
    
    def _create_default_decision(self) -> RoutingDecision:
        """Create default routing decision (hybrid_rerank)."""
        return RoutingDecision(
            strategy=RetrievalStrategy.HYBRID_RERANK,
            complexity=QueryComplexity.MEDIUM,
            confidence=1.0,
            reasoning=["Adaptive routing disabled, using default strategy"],
            estimated_time_ms=1500,
            features_enabled=self.features.copy()
        )
    
    def _update_routing_metrics(self, complexity: QueryComplexity):
        """Update routing decision metrics."""
        self.metrics.total_queries += 1
        
        if complexity == QueryComplexity.SIMPLE:
            self.metrics.simple_queries += 1
        elif complexity == QueryComplexity.MEDIUM:
            self.metrics.medium_queries += 1
        else:
            self.metrics.complex_queries += 1
    
    def _update_execution_metrics(self, execution_time: float):
        """Update execution performance metrics."""
        # Simple exponential moving average
        if self.metrics.avg_retrieval_time_ms == 0:
            self.metrics.avg_retrieval_time_ms = execution_time
        else:
            alpha = 0.1  # Smoothing factor
            self.metrics.avg_retrieval_time_ms = (
                alpha * execution_time + 
                (1 - alpha) * self.metrics.avg_retrieval_time_ms
            )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current routing and performance metrics."""
        return {
            'routing_metrics': {
                'total_queries': self.metrics.total_queries,
                'simple_queries': self.metrics.simple_queries,
                'medium_queries': self.metrics.medium_queries,
                'complex_queries': self.metrics.complex_queries,
                'simple_percentage': (self.metrics.simple_queries / max(1, self.metrics.total_queries)) * 100,
                'complex_percentage': (self.metrics.complex_queries / max(1, self.metrics.total_queries)) * 100
            },
            'performance_metrics': {
                'avg_classification_time_ms': self.metrics.avg_classification_time_ms,
                'avg_retrieval_time_ms': self.metrics.avg_retrieval_time_ms,
                'accuracy_feedback': self.metrics.accuracy_feedback.copy()
            },
            'feature_status': self.features.copy(),
            'performance_targets': self.performance_targets.copy()
        }
    
    def update_feature_flag(self, feature_name: str, enabled: bool):
        """Update feature flag dynamically."""
        if feature_name in self.features:
            self.features[feature_name] = enabled
            self.logger.info(f"Feature flag updated: {feature_name} = {enabled}")
        else:
            self.logger.warning(f"Unknown feature flag: {feature_name}")


# Global adaptive router instance
adaptive_router = AdaptiveRAGRouter()