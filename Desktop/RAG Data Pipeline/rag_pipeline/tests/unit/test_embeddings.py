"""Unit tests for embedding functionality."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from rag_pipeline.vector_store.embeddings import EmbeddingModel
from rag_pipeline.core.exceptions import EmbeddingException


class TestEmbeddingModel:
    """Test suite for EmbeddingModel."""
    
    @pytest.fixture
    def embedding_model(self):
        """Create a test embedding model instance."""
        return EmbeddingModel(model_name="test-model", device="cpu")
    
    @patch('rag_pipeline.vector_store.embeddings.SentenceTransformer')
    def test_model_loading(self, mock_sentence_transformer, embedding_model):
        """Test model loading functionality."""
        # Mock the SentenceTransformer
        mock_model = Mock()
        mock_sentence_transformer.return_value = mock_model
        
        # Load the model
        embedding_model.load()
        
        # Verify model was loaded
        assert embedding_model._model is not None
        mock_sentence_transformer.assert_called_once_with("test-model", device="cpu")
    
    @patch('rag_pipeline.vector_store.embeddings.SentenceTransformer')
    def test_model_loading_failure(self, mock_sentence_transformer, embedding_model):
        """Test model loading failure handling."""
        # Mock SentenceTransformer to raise an exception
        mock_sentence_transformer.side_effect = Exception("Model loading failed")
        
        with pytest.raises(EmbeddingException) as exc_info:
            embedding_model.load()
        
        assert "Failed to load embedding model" in str(exc_info.value)
        assert exc_info.value.error_code == "MODEL_LOAD_FAILED"
    
    @patch('rag_pipeline.vector_store.embeddings.SentenceTransformer')
    def test_embed_single(self, mock_sentence_transformer, embedding_model):
        """Test single text embedding."""
        # Mock the model
        mock_model = Mock()
        mock_embedding = np.array([0.1, 0.2, 0.3, 0.4])
        mock_model.encode.return_value = np.array([mock_embedding])
        mock_sentence_transformer.return_value = mock_model
        
        result = embedding_model.embed_single("test text")
        
        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, mock_embedding)
    
    @patch('rag_pipeline.vector_store.embeddings.SentenceTransformer')
    def test_embed_batch(self, mock_sentence_transformer, embedding_model):
        """Test batch text embedding."""
        # Mock the model
        mock_model = Mock()
        mock_embeddings = np.array([
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
            [0.9, 1.0, 1.1, 1.2]
        ])
        mock_model.encode.return_value = mock_embeddings
        mock_sentence_transformer.return_value = mock_model
        
        texts = ["text 1", "text 2", "text 3"]
        results = embedding_model.embed_batch(texts)
        
        assert len(results) == 3
        assert all(isinstance(emb, np.ndarray) for emb in results)
        
        # Check that preprocessing was applied
        mock_model.encode.assert_called_once()
        call_args = mock_model.encode.call_args[0][0]
        assert all("Represent this document for retrieval:" in text for text in call_args)
    
    def test_embed_empty_list(self, embedding_model):
        """Test embedding empty list."""
        results = embedding_model.embed_batch([])
        assert results == []
    
    def test_embed_empty_text(self, embedding_model):
        """Test embedding empty text."""
        with pytest.raises(EmbeddingException) as exc_info:
            embedding_model.embed_single("")
        
        assert "Empty text provided for embedding" in str(exc_info.value)
        assert exc_info.value.error_code == "EMPTY_TEXT"
    
    def test_preprocess_text(self, embedding_model):
        """Test text preprocessing."""
        original_text = "This is a test."
        processed_text = embedding_model._preprocess_text(original_text)
        
        assert processed_text.startswith("Represent this document for retrieval:")
        assert original_text in processed_text
    
    def test_preprocess_empty_text(self, embedding_model):
        """Test preprocessing empty text."""
        with pytest.raises(EmbeddingException):
            embedding_model._preprocess_text("")
    
    def test_preprocess_whitespace_text(self, embedding_model):
        """Test preprocessing whitespace-only text."""
        with pytest.raises(EmbeddingException):
            embedding_model._preprocess_text("   \n\t   ")
    
    @patch('rag_pipeline.vector_store.embeddings.SentenceTransformer')
    def test_get_embedding_dimension(self, mock_sentence_transformer, embedding_model):
        """Test getting embedding dimension."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_sentence_transformer.return_value = mock_model
        
        dimension = embedding_model.get_embedding_dimension()
        
        assert dimension == 768
        mock_model.get_sentence_embedding_dimension.assert_called_once()
    
    def test_similarity_calculation(self, embedding_model):
        """Test cosine similarity calculation."""
        emb1 = np.array([1.0, 0.0, 0.0])
        emb2 = np.array([0.0, 1.0, 0.0])
        emb3 = np.array([1.0, 0.0, 0.0])
        
        # Orthogonal vectors should have 0 similarity
        similarity = embedding_model.similarity(emb1, emb2)
        assert abs(similarity - 0.0) < 1e-6
        
        # Identical vectors should have 1.0 similarity
        similarity = embedding_model.similarity(emb1, emb3)
        assert abs(similarity - 1.0) < 1e-6
    
    def test_similarity_zero_vector(self, embedding_model):
        """Test similarity with zero vectors."""
        emb1 = np.array([1.0, 0.0, 0.0])
        zero_emb = np.array([0.0, 0.0, 0.0])
        
        similarity = embedding_model.similarity(emb1, zero_emb)
        assert similarity == 0.0
    
    def test_similarity_calculation_error(self, embedding_model):
        """Test similarity calculation with invalid inputs."""
        with pytest.raises(EmbeddingException) as exc_info:
            embedding_model.similarity("invalid", np.array([1.0, 0.0]))
        
        assert "Failed to calculate similarity" in str(exc_info.value)
        assert exc_info.value.error_code == "SIMILARITY_FAILED"
    
    @patch('rag_pipeline.vector_store.embeddings.SentenceTransformer')
    def test_embedding_failure(self, mock_sentence_transformer, embedding_model):
        """Test embedding generation failure."""
        mock_model = Mock()
        mock_model.encode.side_effect = Exception("Encoding failed")
        mock_sentence_transformer.return_value = mock_model
        
        with pytest.raises(EmbeddingException) as exc_info:
            embedding_model.embed_batch(["test text"])
        
        assert "Failed to generate embeddings" in str(exc_info.value)
        assert exc_info.value.error_code == "EMBEDDING_FAILED"
    
    @patch('rag_pipeline.vector_store.embeddings.SentenceTransformer')
    @patch('rag_pipeline.vector_store.embeddings.asyncio')
    async def test_embed_batch_async(self, mock_asyncio, mock_sentence_transformer, embedding_model):
        """Test async batch embedding."""
        # Mock the model
        mock_model = Mock()
        mock_embeddings = np.array([[0.1, 0.2, 0.3]])
        mock_model.encode.return_value = mock_embeddings
        mock_sentence_transformer.return_value = mock_model
        
        # Mock asyncio executor
        mock_loop = Mock()
        mock_asyncio.get_event_loop.return_value = mock_loop
        mock_loop.run_in_executor.return_value = [mock_embeddings[0]]
        
        texts = ["test text"]
        results = await embedding_model.embed_batch_async(texts)
        
        assert len(results) == 1
        mock_loop.run_in_executor.assert_called_once()
    
    def test_thread_safety(self, embedding_model):
        """Test thread-safe model loading."""
        # This test verifies that the RLock is properly used
        # Multiple threads calling load() should not cause issues
        
        import threading
        import time
        
        results = []
        
        @patch('rag_pipeline.vector_store.embeddings.SentenceTransformer')
        def thread_function(mock_st):
            mock_model = Mock()
            mock_st.return_value = mock_model
            try:
                embedding_model.load()
                results.append("success")
            except Exception as e:
                results.append(f"error: {e}")
        
        threads = []
        for i in range(5):
            thread = threading.Thread(target=thread_function)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All threads should succeed, and model should be loaded only once
        assert len(results) == 5
        assert all(result == "success" for result in results)


if __name__ == "__main__":
    pytest.main([__file__])