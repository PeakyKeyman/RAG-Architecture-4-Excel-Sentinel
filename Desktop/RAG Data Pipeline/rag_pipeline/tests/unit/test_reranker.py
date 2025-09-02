"""Unit tests for reranker functionality."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import requests

from rag_pipeline.inference.reranker.cohere_reranker import CohereReranker
from rag_pipeline.inference.reranker.reranker_factory import RerankerFactory
from rag_pipeline.models.reranker import RerankCandidate, RerankResult
from rag_pipeline.core.exceptions import RerankerException


class TestCohereReranker:
    """Test suite for CohereReranker."""
    
    @pytest.fixture
    def reranker(self):
        """Create a test reranker instance."""
        return CohereReranker(api_key="test_api_key", model_name="test-rerank-model")
    
    @pytest.fixture
    def sample_candidates(self):
        """Create sample rerank candidates."""
        return [
            RerankCandidate(
                chunk_id="chunk_1",
                content="This is about artificial intelligence and machine learning.",
                metadata={"source": "doc1"},
                original_score=0.8
            ),
            RerankCandidate(
                chunk_id="chunk_2", 
                content="This document discusses cooking recipes and food preparation.",
                metadata={"source": "doc2"},
                original_score=0.6
            ),
            RerankCandidate(
                chunk_id="chunk_3",
                content="Machine learning algorithms are used in AI systems.",
                metadata={"source": "doc3"},
                original_score=0.7
            )
        ]
    
    def test_initialization_with_api_key(self):
        """Test reranker initialization with API key."""
        reranker = CohereReranker(api_key="test_key")
        assert reranker.api_key == "test_key"
        assert reranker.base_url == "https://api.cohere.ai/v1"
        assert not reranker._initialized
    
    def test_initialization_without_api_key(self):
        """Test reranker initialization without API key."""
        with patch('rag_pipeline.inference.reranker.cohere_reranker.settings') as mock_settings:
            mock_settings.cohere_api_key = None
            
            with pytest.raises(RerankerException) as exc_info:
                CohereReranker()
            
            assert "API key not provided" in str(exc_info.value)
            assert exc_info.value.error_code == "MISSING_API_KEY"
    
    @patch('rag_pipeline.inference.reranker.cohere_reranker.requests.Session')
    def test_initialization_success(self, mock_session, reranker):
        """Test successful reranker initialization."""
        # Mock session and response
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_session_instance.post.return_value = mock_response
        
        reranker.initialize()
        
        assert reranker._initialized
        mock_session.assert_called_once()
        mock_session_instance.post.assert_called_once()
    
    @patch('rag_pipeline.inference.reranker.cohere_reranker.requests.Session')
    def test_initialization_api_failure(self, mock_session, reranker):
        """Test reranker initialization with API failure."""
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance
        
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        mock_session_instance.post.return_value = mock_response
        
        with pytest.raises(RerankerException) as exc_info:
            reranker.initialize()
        
        assert "API test failed" in str(exc_info.value)
        assert exc_info.value.error_code == "COHERE_API_ERROR"
    
    @patch('rag_pipeline.inference.reranker.cohere_reranker.requests.Session')
    def test_rerank_success(self, mock_session, reranker, sample_candidates):
        """Test successful reranking."""
        # Setup mock session
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance
        
        # Mock initialization response
        init_response = Mock()
        init_response.status_code = 200
        
        # Mock rerank response
        rerank_response = Mock()
        rerank_response.status_code = 200
        rerank_response.json.return_value = {
            "results": [
                {"index": 2, "relevance_score": 0.95},  # chunk_3 most relevant
                {"index": 0, "relevance_score": 0.85},  # chunk_1 second
                {"index": 1, "relevance_score": 0.25},  # chunk_2 least relevant
            ]
        }
        
        mock_session_instance.post.side_effect = [init_response, rerank_response]
        
        query = "machine learning algorithms"
        results = reranker.rerank(query, sample_candidates, top_k=3)
        
        assert len(results) == 3
        assert all(isinstance(r, RerankResult) for r in results)
        
        # Check ordering (should be sorted by relevance)
        assert results[0].chunk_id == "chunk_3"  # Highest score
        assert results[1].chunk_id == "chunk_1"  # Second highest
        assert results[2].chunk_id == "chunk_2"  # Lowest score
        
        # Check scores
        assert results[0].score == 0.95
        assert results[1].score == 0.85
        assert results[2].score == 0.25
    
    def test_rerank_empty_candidates(self, reranker):
        """Test reranking with empty candidates."""
        results = reranker.rerank("test query", [], top_k=5)
        assert results == []
    
    def test_validate_candidates(self, reranker, sample_candidates):
        """Test candidate validation."""
        # Add invalid candidates
        invalid_candidates = sample_candidates + [
            RerankCandidate(chunk_id="", content="valid content", metadata={}),
            RerankCandidate(chunk_id="valid_id", content="", metadata={}),
            RerankCandidate(chunk_id="valid_id2", content="   ", metadata={})
        ]
        
        valid_candidates = reranker.validate_candidates(invalid_candidates)
        
        # Should only return the original 3 valid candidates
        assert len(valid_candidates) == 3
        assert all(c.chunk_id in ["chunk_1", "chunk_2", "chunk_3"] for c in valid_candidates)
    
    @patch('rag_pipeline.inference.reranker.cohere_reranker.requests.Session')
    def test_rerank_api_failure(self, mock_session, reranker, sample_candidates):
        """Test reranking with API failure."""
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance
        
        # Mock successful initialization
        init_response = Mock()
        init_response.status_code = 200
        
        # Mock failed rerank response
        rerank_response = Mock()
        rerank_response.status_code = 500
        rerank_response.text = "Internal Server Error"
        
        mock_session_instance.post.side_effect = [init_response, rerank_response]
        
        with pytest.raises(RerankerException) as exc_info:
            reranker.rerank("test query", sample_candidates, top_k=2)
        
        assert "Cohere rerank failed" in str(exc_info.value)
        assert exc_info.value.error_code == "COHERE_RERANK_FAILED"
    
    @patch('rag_pipeline.inference.reranker.cohere_reranker.requests.Session')
    def test_thread_safety(self, mock_session, reranker):
        """Test thread-safe initialization."""
        import threading
        
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_session_instance.post.return_value = mock_response
        
        results = []
        
        def init_reranker():
            try:
                reranker.initialize()
                results.append("success")
            except Exception as e:
                results.append(f"error: {e}")
        
        threads = []
        for i in range(5):
            thread = threading.Thread(target=init_reranker)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All threads should succeed, initialization should happen only once
        assert len(results) == 5
        assert all(result == "success" for result in results)
        assert reranker._initialized
    
    def test_get_model_info(self, reranker):
        """Test getting model information."""
        info = reranker.get_model_info()
        
        assert info["model_name"] == "test-rerank-model"
        assert info["provider"] == "cohere"
        assert info["reranker_type"] == "CohereReranker"
        assert info["has_api_key"] is True


class TestRerankerFactory:
    """Test suite for RerankerFactory."""
    
    def test_create_cohere_reranker(self):
        """Test creating Cohere reranker through factory."""
        with patch('rag_pipeline.inference.reranker.cohere_reranker.CohereReranker') as mock_cohere:
            mock_instance = Mock()
            mock_cohere.return_value = mock_instance
            
            reranker = RerankerFactory.create_reranker(
                reranker_type="cohere",
                api_key="test_key",
                model_name="test-model"
            )
            
            assert reranker == mock_instance
            mock_cohere.assert_called_once_with(
                api_key="test_key",
                model_name="test-model"
            )
    
    def test_create_unknown_reranker(self):
        """Test creating unknown reranker type."""
        with pytest.raises(RerankerException) as exc_info:
            RerankerFactory.create_reranker(reranker_type="unknown")
        
        assert "Unknown reranker type" in str(exc_info.value)
        assert exc_info.value.error_code == "UNKNOWN_RERANKER_TYPE"
    
    def test_get_available_rerankers(self):
        """Test getting available reranker types."""
        available = RerankerFactory.get_available_rerankers()
        
        assert isinstance(available, dict)
        assert "cohere" in available
        assert isinstance(available["cohere"], str)
    
    def test_create_default_reranker(self):
        """Test creating default reranker."""
        with patch('rag_pipeline.inference.reranker.reranker_factory.settings') as mock_settings:
            mock_settings.reranker_type = "cohere"
            mock_settings.cohere_api_key = "test_key"
            
            with patch('rag_pipeline.inference.reranker.cohere_reranker.CohereReranker') as mock_cohere:
                mock_instance = Mock()
                mock_cohere.return_value = mock_instance
                
                from rag_pipeline.inference.reranker.reranker_factory import create_default_reranker
                reranker = create_default_reranker()
                
                assert reranker == mock_instance
    
    def test_register_new_reranker(self):
        """Test registering a new reranker type."""
        from rag_pipeline.inference.reranker.base_reranker import BaseReranker
        
        class TestReranker(BaseReranker):
            def _initialize(self):
                pass
            
            def _rerank_batch(self, query, candidates, top_k):
                return []
        
        RerankerFactory.register_reranker("test_reranker", TestReranker)
        
        available = RerankerFactory.get_available_rerankers()
        assert "test_reranker" in available
        
        # Clean up
        if "test_reranker" in RerankerFactory._rerankers:
            del RerankerFactory._rerankers["test_reranker"]
    
    def test_register_invalid_reranker(self):
        """Test registering invalid reranker class."""
        class InvalidReranker:
            pass
        
        with pytest.raises(RerankerException) as exc_info:
            RerankerFactory.register_reranker("invalid", InvalidReranker)
        
        assert "must inherit from BaseReranker" in str(exc_info.value)
        assert exc_info.value.error_code == "INVALID_RERANKER_CLASS"


if __name__ == "__main__":
    pytest.main([__file__])