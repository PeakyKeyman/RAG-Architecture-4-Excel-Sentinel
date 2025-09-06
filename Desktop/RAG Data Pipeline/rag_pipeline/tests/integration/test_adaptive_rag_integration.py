"""Integration tests for Adaptive RAG and Temporal Understanding features."""

import pytest
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock, AsyncMock

from ...inference.pipeline import InferencePipeline
from ...inference.adaptive_router import adaptive_router, RetrievalStrategy
from ...parsing.document_parser import DocumentParser
from ...parsing.temporal_analyzer import temporal_analyzer


class TestAdaptiveRAGIntegration:
    """Integration tests for adaptive RAG system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.pipeline = InferencePipeline()
        
    @pytest.mark.asyncio
    async def test_adaptive_pipeline_simple_query(self):
        """Test that simple queries use optimized routing through the pipeline."""
        simple_query = "What is revenue?"
        
        with patch.object(self.pipeline, 'reranker') as mock_reranker, \
             patch('...inference.pipeline.adaptive_router') as mock_router, \
             patch('...inference.pipeline.context_packager') as mock_packager:
            
            # Mock adaptive router routing decision
            mock_decision = MagicMock()
            mock_decision.strategy = RetrievalStrategy.DIRECT_VECTOR
            mock_decision.complexity.value = "simple"
            mock_decision.confidence = 0.95
            mock_router.route_query.return_value = mock_decision
            
            # Mock strategy execution  
            mock_results = {
                'chunks': [
                    {'chunk_id': 'test1', 'content': 'Revenue is income', 'score': 0.9}
                ],
                'strategy': 'direct_vector',
                'temporal_scoring_applied': True,
                'routing_metadata': {
                    'strategy': 'direct_vector',
                    'execution_time_ms': 300
                }
            }
            mock_router.execute_strategy = AsyncMock(return_value=mock_results)
            
            # Mock context packaging
            mock_context = MagicMock()
            mock_context.contexts = ["Revenue is income"]
            mock_context.total_tokens = 50
            mock_packager.package_context.return_value = mock_context
            
            # Execute pipeline
            response = await self.pipeline._execute_pipeline(
                MagicMock(
                    request_id="test123",
                    query=simple_query,
                    top_k=5,
                    filters=None,
                    metadata={'org_id': 'test_org'},
                    user_id='test_user'
                )
            )
            
            # Verify adaptive routing was used
            mock_router.route_query.assert_called_once()
            mock_router.execute_strategy.assert_called_once()
            
            # Verify response structure
            assert response.success is True
            assert 'selected_strategy' in response.metadata
            assert response.metadata['selected_strategy'] == 'direct_vector'
    
    @pytest.mark.asyncio
    async def test_adaptive_pipeline_complex_query(self):
        """Test that complex queries use full ensemble routing."""
        complex_query = "Analyze our Q3 performance compared to Q2 and recommend strategic initiatives"
        
        with patch.object(self.pipeline, 'reranker') as mock_reranker, \
             patch('...inference.pipeline.adaptive_router') as mock_router, \
             patch('...inference.pipeline.context_packager') as mock_packager:
            
            # Mock complex routing decision
            mock_decision = MagicMock()
            mock_decision.strategy = RetrievalStrategy.FULL_ENSEMBLE
            mock_decision.complexity.value = "complex"
            mock_decision.confidence = 0.88
            mock_router.route_query.return_value = mock_decision
            
            # Mock ensemble execution results
            mock_results = {
                'chunks': [
                    {'chunk_id': f'chunk_{i}', 'content': f'Analysis point {i}', 'score': 0.9-i*0.1}
                    for i in range(5)
                ],
                'strategy': 'full_ensemble',
                'hyde_queries': 3,
                'temporal_scoring_applied': True,
                'routing_metadata': {
                    'strategy': 'full_ensemble',
                    'execution_time_ms': 4500
                }
            }
            mock_router.execute_strategy = AsyncMock(return_value=mock_results)
            
            # Mock context packaging
            mock_context = MagicMock()
            mock_context.contexts = [f"Analysis point {i}" for i in range(5)]
            mock_context.total_tokens = 500
            mock_packager.package_context.return_value = mock_context
            
            # Execute pipeline
            response = await self.pipeline._execute_pipeline(
                MagicMock(
                    request_id="test456", 
                    query=complex_query,
                    top_k=10,
                    filters=None,
                    metadata={'org_id': 'test_org'},
                    user_id='test_user'
                )
            )
            
            # Verify ensemble strategy was used
            mock_router.route_query.assert_called_once()
            mock_router.execute_strategy.assert_called_once()
            
            # Verify response indicates complex processing
            assert response.success is True
            assert response.metadata['selected_strategy'] == 'full_ensemble'
            assert 'hyde_queries' in response.metadata.get('routing_metadata', {})


class TestTemporalIntegration:
    """Integration tests for temporal understanding features."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = DocumentParser()
    
    def test_document_parsing_with_temporal_analysis(self):
        """Test that document parsing includes temporal metadata extraction."""
        test_content = """
        Quarterly Earnings Report
        Q3 2024 Results
        Date: September 30, 2024
        
        Financial Performance:
        - Revenue: $15.2M (up 18% year-over-year from Q3 2023)
        - Operating margin: 24.5% (vs 22.1% last quarter)
        - Current cash position: $45.8M as of September 30, 2024
        """
        
        with patch.object(self.parser, '_ensure_local_file') as mock_local, \
             patch('builtins.open', create=True) as mock_open, \
             patch('...parsing.document_parser.pdfplumber') as mock_pdf:
            
            # Mock file operations
            mock_local.return_value = "/tmp/test.pdf"
            
            # Mock PDF extraction
            mock_page = MagicMock()
            mock_page.extract_text.return_value = test_content
            mock_page.extract_tables.return_value = []
            
            mock_pdf_obj = MagicMock()
            mock_pdf_obj.pages = [mock_page]
            mock_pdf_obj.metadata = {'Creator': 'Test'}
            mock_pdf_obj.__enter__.return_value = mock_pdf_obj
            
            mock_pdf.open.return_value = mock_pdf_obj
            
            # Parse document
            document = self.parser.parse_document(
                file_path="/tmp/test.pdf",
                user_id="test_user",
                group_id="test_group", 
                org_id="test_org",
                document_metadata={'category': 'financial'}
            )
            
            # Verify temporal metadata was extracted
            metadata = document.metadata
            
            assert 'document_date' in metadata
            assert 'temporal_relevance' in metadata
            assert 'recency_score' in metadata
            assert 'fiscal_quarter' in metadata
            assert 'fiscal_year' in metadata
            
            # Verify financial document was classified correctly
            assert metadata['document_type'] == 'financial'
            
            # Verify temporal relevance is current (recent document)
            assert metadata['temporal_relevance'] in ['current', 'recent']
            assert metadata['recency_score'] > 0.5
    
    def test_temporal_query_analysis_integration(self):
        """Test temporal query analysis integration."""
        test_cases = [
            {
                'query': "What were our current quarterly results?",
                'expected_temporal_intent': True,
                'expected_preference': 'current'
            },
            {
                'query': "Compare this year's performance to last year",
                'expected_temporal_intent': True,
                'expected_preference': 'current'
            },
            {
                'query': "Historical trends in market share",
                'expected_temporal_intent': True,
                'expected_preference': 'historical'  
            },
            {
                'query': "What is machine learning?",
                'expected_temporal_intent': False,
                'expected_preference': 'current'
            }
        ]
        
        for test_case in test_cases:
            analysis = temporal_analyzer.analyze_query_temporality(test_case['query'])
            
            assert analysis['has_temporal_intent'] == test_case['expected_temporal_intent']
            assert analysis['recency_preference'] == test_case['expected_preference']


class TestEndToEndAdaptiveTemporal:
    """End-to-end integration tests combining adaptive RAG and temporal features."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.pipeline = InferencePipeline()
    
    @pytest.mark.asyncio
    async def test_temporal_query_adaptive_routing(self):
        """Test temporal queries get appropriate routing and scoring."""
        temporal_query = "What changed in our financial performance since last quarter?"
        
        with patch.object(self.pipeline, 'reranker') as mock_reranker, \
             patch('...inference.pipeline.adaptive_router') as mock_router, \
             patch('...inference.pipeline.context_packager') as mock_packager:
            
            # Mock routing decision for temporal query
            mock_decision = MagicMock()
            mock_decision.strategy = RetrievalStrategy.HYBRID_RERANK  # Temporal queries often medium complexity
            mock_decision.complexity.value = "medium"
            mock_decision.confidence = 0.85
            mock_router.route_query.return_value = mock_decision
            
            # Mock results with temporal scoring
            mock_results = {
                'chunks': [
                    {
                        'chunk_id': 'financial_q3_2024',
                        'content': 'Q3 2024 revenue increased 15% from Q2 2024',
                        'score': 0.9,
                        'temporal_score': 1.5,  # High temporal relevance
                        'combined_score': 1.35,
                        'metadata': {
                            'document_date': '2024-09-30',
                            'temporal_relevance': 'current',
                            'fiscal_quarter': 'Q3',
                            'document_type': 'financial'
                        }
                    },
                    {
                        'chunk_id': 'financial_q2_2024',
                        'content': 'Q2 2024 revenue was $12.5M',
                        'score': 0.85,
                        'temporal_score': 1.2,  # Good temporal relevance
                        'combined_score': 1.02,
                        'metadata': {
                            'document_date': '2024-06-30',
                            'temporal_relevance': 'recent',
                            'fiscal_quarter': 'Q2',
                            'document_type': 'financial'
                        }
                    }
                ],
                'strategy': 'hybrid_rerank',
                'temporal_scoring_applied': True,
                'routing_metadata': {
                    'strategy': 'hybrid_rerank',
                    'query_temporal_match': True
                }
            }
            mock_router.execute_strategy = AsyncMock(return_value=mock_results)
            
            # Mock context packaging
            mock_context = MagicMock()
            mock_context.contexts = [chunk['content'] for chunk in mock_results['chunks']]
            mock_context.total_tokens = 150
            mock_packager.package_context.return_value = mock_context
            
            # Execute pipeline
            response = await self.pipeline._execute_pipeline(
                MagicMock(
                    request_id="temporal_test",
                    query=temporal_query,
                    top_k=5,
                    filters=None,
                    metadata={'org_id': 'test_org'},
                    user_id='test_user'
                )
            )
            
            # Verify temporal features were applied
            assert response.success is True
            assert response.metadata['selected_strategy'] == 'hybrid_rerank'
            
            routing_metadata = response.metadata.get('routing_metadata', {})
            assert routing_metadata.get('temporal_scoring_applied') is True
    
    def test_performance_meets_targets(self):
        """Test that adaptive routing meets performance targets."""
        # Mock performance monitoring
        with patch.object(adaptive_router, 'logger') as mock_logger:
            
            # Test simple query performance target
            simple_decision = adaptive_router.route_query("What is AI?")
            assert simple_decision.estimated_time_ms <= adaptive_router.performance_targets.get('simple_query_max_ms', 500)
            
            # Test medium query performance target
            medium_decision = adaptive_router.route_query("Explain our business strategy")
            assert medium_decision.estimated_time_ms <= adaptive_router.performance_targets.get('medium_query_max_ms', 2000)
            
            # Test complex query performance target
            complex_decision = adaptive_router.route_query("Analyze market trends and competitive positioning with strategic recommendations")
            assert complex_decision.estimated_time_ms <= adaptive_router.performance_targets.get('complex_query_max_ms', 8000)
    
    def test_feature_flag_integration(self):
        """Test that feature flags properly control system behavior."""
        
        # Test with adaptive routing disabled
        adaptive_router.update_feature_flag('adaptive_routing_enabled', False)
        
        decision = adaptive_router.route_query("Any query")
        assert decision.strategy == RetrievalStrategy.HYBRID_RERANK  # Should use default
        
        # Re-enable for other tests
        adaptive_router.update_feature_flag('adaptive_routing_enabled', True)
        
        # Test with simple query optimization disabled
        adaptive_router.update_feature_flag('simple_query_optimization', False)
        
        decision = adaptive_router.route_query("What is AI?")  # Simple query
        assert decision.strategy == RetrievalStrategy.HYBRID_RERANK  # Should not use direct vector
        
        # Re-enable
        adaptive_router.update_feature_flag('simple_query_optimization', True)
    
    def test_graceful_degradation(self):
        """Test system graceful degradation when components fail."""
        
        # Test adaptive router fallback
        with patch('...inference.adaptive_router.query_classifier') as mock_classifier:
            mock_classifier.classify.side_effect = Exception("Classifier failed")
            
            # Should not crash, should fall back to default
            decision = adaptive_router.route_query("test query")
            assert decision.strategy == RetrievalStrategy.HYBRID_RERANK
            assert "disabled" in decision.reasoning[0].lower()
    
    def test_metrics_collection(self):
        """Test that performance metrics are properly collected."""
        # Reset metrics
        adaptive_router.metrics = adaptive_router.RoutingMetrics()
        
        # Route various queries
        test_queries = [
            "What is AI?",  # Simple
            "Explain our strategy",  # Medium
            "Analyze competitive landscape with recommendations",  # Complex
            "Current revenue?",  # Simple
        ]
        
        with patch('...inference.adaptive_router.query_classifier') as mock_classifier:
            complexities = ['simple', 'medium', 'complex', 'simple']
            
            for i, query in enumerate(test_queries):
                mock_classification = MagicMock()
                mock_classification.complexity.value = complexities[i]
                mock_classification.confidence = 0.9
                mock_classification.reasoning = ["test"]
                mock_classifier.classify.return_value = mock_classification
                
                adaptive_router.route_query(query)
        
        # Check metrics
        metrics = adaptive_router.get_metrics()
        routing_metrics = metrics['routing_metrics']
        
        assert routing_metrics['total_queries'] == 4
        assert routing_metrics['simple_queries'] == 2
        assert routing_metrics['medium_queries'] == 1
        assert routing_metrics['complex_queries'] == 1
        assert routing_metrics['simple_percentage'] == 50.0


class TestSystemIntegrationHealthCheck:
    """Health check tests for the integrated system."""
    
    def test_all_components_importable(self):
        """Test that all new components can be imported without errors."""
        try:
            from ...inference.query_classifier import ExecutiveQueryClassifier
            from ...inference.adaptive_router import AdaptiveRAGRouter
            from ...inference.temporal_scorer import TemporalScorer
            from ...parsing.temporal_analyzer import TemporalAnalyzer
            
            # Try to instantiate core components
            classifier = ExecutiveQueryClassifier()
            router = AdaptiveRAGRouter()
            scorer = TemporalScorer()
            analyzer = TemporalAnalyzer()
            
            assert classifier is not None
            assert router is not None
            assert scorer is not None
            assert analyzer is not None
            
        except ImportError as e:
            pytest.fail(f"Import error: {str(e)}")
        except Exception as e:
            pytest.fail(f"Component instantiation error: {str(e)}")
    
    def test_configuration_loading(self):
        """Test that configuration is properly loaded for new features."""
        # Test that adaptive router can be configured
        config = {
            'adaptive_routing_enabled': True,
            'simple_query_optimization': True,
            'performance_monitoring': True
        }
        
        router = AdaptiveRAGRouter(config)
        assert router.features['adaptive_routing_enabled'] is True
        assert router.features['simple_query_optimization'] is True
        
        # Test that temporal analyzer can be configured
        temporal_config = {
            'fiscal_year_start': 'April',
            'relevance_decay_months': 18
        }
        
        analyzer = temporal_analyzer.__class__(temporal_config)
        assert analyzer.current_fiscal_year_start == 'April'
        assert analyzer.relevance_decay_months == 18