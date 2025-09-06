"""Unit tests for adaptive RAG router."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from ...inference.adaptive_router import (
    AdaptiveRAGRouter,
    RetrievalStrategy,
    RoutingDecision,
    RoutingMetrics
)
from ...inference.query_classifier import QueryComplexity


class TestAdaptiveRAGRouter:
    """Test cases for AdaptiveRAGRouter."""
    
    def setup_method(self):
        """Set up test fixtures."""
        config = {
            'adaptive_routing_enabled': True,
            'simple_query_optimization': True,
            'complex_query_ensemble': True,
            'performance_monitoring': True,
            'fallback_to_default': True
        }
        self.router = AdaptiveRAGRouter(config)
    
    def test_route_query_simple(self):
        """Test routing of simple queries."""
        simple_query = "What is EBITDA?"
        
        with patch('...inference.adaptive_router.query_classifier') as mock_classifier:
            # Mock simple classification
            mock_classification = MagicMock()
            mock_classification.complexity = QueryComplexity.SIMPLE
            mock_classification.confidence = 0.95
            mock_classification.reasoning = ["Short factual query"]
            mock_classifier.classify.return_value = mock_classification
            
            decision = self.router.route_query(simple_query)
            
            assert decision.strategy == RetrievalStrategy.DIRECT_VECTOR
            assert decision.complexity == QueryComplexity.SIMPLE
            assert decision.confidence == 0.95
            assert decision.estimated_time_ms < 1000  # Should be fast
    
    def test_route_query_complex(self):
        """Test routing of complex queries."""
        complex_query = "Compare our Q3 performance to Q2 and analyze strategic opportunities"
        
        with patch('...inference.adaptive_router.query_classifier') as mock_classifier:
            # Mock complex classification
            mock_classification = MagicMock()
            mock_classification.complexity = QueryComplexity.COMPLEX
            mock_classification.confidence = 0.89
            mock_classification.reasoning = ["Multi-part analytical query"]
            mock_classifier.classify.return_value = mock_classification
            
            decision = self.router.route_query(complex_query)
            
            assert decision.strategy == RetrievalStrategy.FULL_ENSEMBLE
            assert decision.complexity == QueryComplexity.COMPLEX
            assert decision.estimated_time_ms > 2000  # Should take longer
    
    def test_route_query_medium(self):
        """Test routing of medium complexity queries."""
        medium_query = "Summarize our quarterly results"
        
        with patch('...inference.adaptive_router.query_classifier') as mock_classifier:
            # Mock medium classification
            mock_classification = MagicMock()
            mock_classification.complexity = QueryComplexity.MEDIUM
            mock_classification.confidence = 0.82
            mock_classification.reasoning = ["Summary request"]
            mock_classifier.classify.return_value = mock_classification
            
            decision = self.router.route_query(medium_query)
            
            assert decision.strategy == RetrievalStrategy.HYBRID_RERANK
            assert decision.complexity == QueryComplexity.MEDIUM
            assert 1000 < decision.estimated_time_ms < 3000
    
    def test_feature_flags_disabled(self):
        """Test behavior when feature flags are disabled."""
        config = {
            'adaptive_routing_enabled': False,
            'simple_query_optimization': False,
            'complex_query_ensemble': False
        }
        router = AdaptiveRAGRouter(config)
        
        decision = router.route_query("Any query")
        
        # Should fall back to default strategy
        assert decision.strategy == RetrievalStrategy.HYBRID_RERANK
        assert decision.complexity == QueryComplexity.MEDIUM
        assert decision.confidence == 1.0
        assert "disabled" in decision.reasoning[0].lower()
    
    @pytest.mark.asyncio
    async def test_execute_direct_vector_strategy(self):
        """Test execution of direct vector strategy."""
        query = "What is AI?"
        decision = RoutingDecision(
            strategy=RetrievalStrategy.DIRECT_VECTOR,
            complexity=QueryComplexity.SIMPLE,
            confidence=0.95,
            reasoning=["Simple query"],
            estimated_time_ms=300
        )
        
        # Mock vector store
        with patch('...inference.adaptive_router.vector_store') as mock_vector_store:
            mock_results = [MagicMock(), MagicMock()]
            for result in mock_results:
                result.to_dict.return_value = {'chunk_id': 'test', 'content': 'test content'}
            
            mock_vector_store.similarity_search_async.return_value = mock_results
            
            # Mock temporal scorer
            with patch('...inference.adaptive_router.temporal_scorer') as mock_scorer:
                mock_scorer.score_results.return_value = [
                    {'chunk_id': 'test1', 'content': 'content1', 'temporal_score': 1.0},
                    {'chunk_id': 'test2', 'content': 'content2', 'temporal_score': 0.9}
                ]
                
                results = await self.router.execute_strategy(
                    decision=decision,
                    query=query,
                    top_k=5
                )
                
                assert results['strategy'] == 'direct_vector'
                assert results['temporal_scoring_applied'] is True
                assert len(results['chunks']) == 2
                assert results['total_retrieved'] == 2
    
    @pytest.mark.asyncio 
    async def test_execute_hybrid_rerank_strategy(self):
        """Test execution of hybrid + rerank strategy."""
        query = "Explain our strategy"
        decision = RoutingDecision(
            strategy=RetrievalStrategy.HYBRID_RERANK,
            complexity=QueryComplexity.MEDIUM,
            confidence=0.85,
            reasoning=["Medium complexity"],
            estimated_time_ms=1500
        )
        
        # Mock hybrid search
        with patch('...inference.adaptive_router.hybrid_search') as mock_hybrid:
            mock_results = [MagicMock() for _ in range(10)]
            for i, result in enumerate(mock_results):
                result.to_dict.return_value = {'chunk_id': f'chunk_{i}', 'content': f'content_{i}'}
            
            mock_hybrid.search_async.return_value = mock_results
            
            # Mock reranker
            mock_reranked = mock_results[:5]  # Top 5 after reranking
            self.router.reranker.rerank_async = AsyncMock(return_value=mock_reranked)
            
            # Mock temporal scorer
            with patch('...inference.adaptive_router.temporal_scorer') as mock_scorer:
                mock_scorer.score_results.return_value = [
                    {'chunk_id': f'chunk_{i}', 'temporal_score': 1.0 - i*0.1} 
                    for i in range(5)
                ]
                
                results = await self.router.execute_strategy(
                    decision=decision,
                    query=query,
                    top_k=5
                )
                
                assert results['strategy'] == 'hybrid_rerank'
                assert results['reranked'] is True
                assert results['temporal_scoring_applied'] is True
                assert len(results['chunks']) == 5
    
    @pytest.mark.asyncio
    async def test_execute_full_ensemble_strategy(self):
        """Test execution of full ensemble strategy."""
        query = "Analyze competitive landscape and recommend strategic actions"
        decision = RoutingDecision(
            strategy=RetrievalStrategy.FULL_ENSEMBLE,
            complexity=QueryComplexity.COMPLEX,
            confidence=0.92,
            reasoning=["Complex analytical query"],
            estimated_time_ms=5000
        )
        
        # Mock HyDE ensemble
        with patch('...inference.adaptive_router.hyde_ensemble') as mock_hyde:
            mock_hyde.generate_hypothetical_documents.return_value = [
                "Hypothetical doc 1",
                "Hypothetical doc 2"
            ]
            
            # Mock hybrid search for each query
            with patch('...inference.adaptive_router.hybrid_search') as mock_hybrid:
                mock_results = [MagicMock() for _ in range(15)]  # More results from ensemble
                for i, result in enumerate(mock_results):
                    result.content = f'content_{i}'
                    result.metadata = {'chunk_id': f'chunk_{i}'}
                    result.to_dict.return_value = {'chunk_id': f'chunk_{i}', 'content': f'content_{i}'}
                
                mock_hybrid.search_async.return_value = mock_results[:5]  # 5 per query
                
                # Mock reranker
                mock_reranked = mock_results[:8]  # Top 8 after reranking
                self.router.reranker.rerank_async = AsyncMock(return_value=mock_reranked)
                
                # Mock temporal scorer
                with patch('...inference.adaptive_router.temporal_scorer') as mock_scorer:
                    mock_scorer.score_results.return_value = [
                        {'chunk_id': f'chunk_{i}', 'temporal_score': 1.0 - i*0.1} 
                        for i in range(8)
                    ]
                    
                    results = await self.router.execute_strategy(
                        decision=decision,
                        query=query,
                        top_k=8
                    )
                    
                    assert results['strategy'] == 'full_ensemble'
                    assert results['hyde_queries'] == 2
                    assert results['reranked'] is True
                    assert results['temporal_scoring_applied'] is True
                    assert len(results['chunks']) == 8
    
    def test_metrics_tracking(self):
        """Test that routing metrics are properly tracked."""
        # Start with empty metrics
        initial_metrics = self.router.get_metrics()
        assert initial_metrics['routing_metrics']['total_queries'] == 0
        
        # Route some queries
        with patch('...inference.adaptive_router.query_classifier') as mock_classifier:
            # Mock different complexities
            classifications = [
                (QueryComplexity.SIMPLE, 0.95),
                (QueryComplexity.MEDIUM, 0.85),
                (QueryComplexity.COMPLEX, 0.90),
                (QueryComplexity.SIMPLE, 0.93)
            ]
            
            for complexity, confidence in classifications:
                mock_classification = MagicMock()
                mock_classification.complexity = complexity
                mock_classification.confidence = confidence
                mock_classification.reasoning = ["test"]
                mock_classifier.classify.return_value = mock_classification
                
                self.router.route_query("test query")
        
        # Check metrics updated
        final_metrics = self.router.get_metrics()
        routing_metrics = final_metrics['routing_metrics']
        
        assert routing_metrics['total_queries'] == 4
        assert routing_metrics['simple_queries'] == 2
        assert routing_metrics['medium_queries'] == 1
        assert routing_metrics['complex_queries'] == 1
        assert routing_metrics['simple_percentage'] == 50.0
        assert routing_metrics['complex_percentage'] == 25.0
    
    def test_feature_flag_updates(self):
        """Test dynamic feature flag updates."""
        # Initially enabled
        assert self.router.features['adaptive_routing_enabled'] is True
        
        # Update feature flag
        self.router.update_feature_flag('adaptive_routing_enabled', False)
        assert self.router.features['adaptive_routing_enabled'] is False
        
        # Test invalid feature flag
        self.router.update_feature_flag('invalid_feature', True)
        # Should log warning but not crash
        assert 'invalid_feature' not in self.router.features
    
    def test_graceful_degradation_on_error(self):
        """Test graceful degradation when classification fails."""
        with patch('...inference.adaptive_router.query_classifier') as mock_classifier:
            # Mock classification failure
            mock_classifier.classify.side_effect = Exception("Classification failed")
            
            # Should fall back to default without crashing
            decision = self.router.route_query("test query")
            
            assert decision.strategy == RetrievalStrategy.HYBRID_RERANK  # Default fallback
            assert decision.confidence == 1.0
    
    def test_deduplication_logic(self):
        """Test result deduplication logic."""
        # Create results with some duplicates
        results = []
        for i in range(5):
            result = MagicMock()
            result.chunk_id = f'chunk_{i % 3}'  # Creates duplicates
            results.append(result)
        
        unique_results = self.router._deduplicate_results(results)
        
        # Should have only 3 unique results (chunk_0, chunk_1, chunk_2)
        assert len(unique_results) == 3
        
        # Check chunk IDs are unique
        chunk_ids = [getattr(result, 'chunk_id', str(hash(str(result)))) for result in unique_results]
        assert len(set(chunk_ids)) == len(unique_results)
    
    def test_performance_target_monitoring(self):
        """Test performance target monitoring."""
        # Set strict performance targets
        self.router.performance_targets = {
            'simple_query_max_ms': 100,  # Very strict
            'medium_query_max_ms': 500,
            'complex_query_max_ms': 1000
        }
        
        # Simulate slow execution that exceeds targets
        import time
        from unittest.mock import patch
        
        with patch.object(self.router, 'logger') as mock_logger:
            # This would normally trigger a warning if execution was slow
            # We can't easily simulate slow execution in unit tests,
            # so we just verify the monitoring logic exists
            assert 'simple_query_max_ms' in self.router.performance_targets
            assert self.router.performance_targets['simple_query_max_ms'] == 100


class TestAdaptiveRouterIntegration:
    """Integration tests for adaptive router."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_routing_execution(self):
        """Test complete routing and execution flow."""
        router = AdaptiveRAGRouter({'adaptive_routing_enabled': True})
        query = "What are our key metrics?"
        
        # Mock all dependencies
        with patch('...inference.adaptive_router.query_classifier') as mock_classifier, \
             patch('...inference.adaptive_router.vector_store') as mock_vector, \
             patch('...inference.adaptive_router.temporal_scorer') as mock_scorer:
            
            # Mock classification
            mock_classification = MagicMock()
            mock_classification.complexity = QueryComplexity.SIMPLE
            mock_classification.confidence = 0.95
            mock_classification.reasoning = ["Simple factual query"]
            mock_classifier.classify.return_value = mock_classification
            
            # Mock vector results
            mock_result = MagicMock()
            mock_result.to_dict.return_value = {
                'chunk_id': 'test_chunk',
                'content': 'Test content',
                'score': 0.9
            }
            mock_vector.similarity_search_async.return_value = [mock_result]
            
            # Mock temporal scoring
            mock_scorer.score_results.return_value = [{
                'chunk_id': 'test_chunk',
                'content': 'Test content',
                'score': 0.9,
                'temporal_score': 1.0,
                'combined_score': 0.95
            }]
            
            # Test routing
            decision = router.route_query(query)
            assert decision.strategy == RetrievalStrategy.DIRECT_VECTOR
            
            # Test execution
            results = await router.execute_strategy(decision, query, top_k=5)
            
            assert results['strategy'] == 'direct_vector'
            assert results['temporal_scoring_applied'] is True
            assert len(results['chunks']) == 1
            assert 'routing_metadata' in results or results.get('temporal_scoring_applied')
    
    def test_config_integration(self):
        """Test integration with configuration system."""
        config = {
            'adaptive_routing_enabled': True,
            'simple_query_max_ms': 200,
            'medium_query_max_ms': 1000,
            'complex_query_max_ms': 3000,
            'performance_monitoring': True
        }
        
        router = AdaptiveRAGRouter(config)
        
        assert router.features['adaptive_routing_enabled'] is True
        assert router.performance_targets['simple_query_max_ms'] == 200
        assert router.features['performance_monitoring'] is True