"""Unit tests for hierarchical chunking functionality."""

import pytest
import uuid
from unittest.mock import Mock, patch

from rag_pipeline.chunking.hierarchical_chunker import HierarchicalChunker
from rag_pipeline.models.chunk import Document, Chunk
from rag_pipeline.core.exceptions import ChunkingException


class TestHierarchicalChunker:
    """Test suite for HierarchicalChunker."""
    
    @pytest.fixture
    def chunker(self):
        """Create a test chunker instance."""
        return HierarchicalChunker(
            child_chunk_size=200,
            parent_chunk_size=600,
            chunk_overlap=0.1
        )
    
    @pytest.fixture
    def sample_document(self):
        """Create a sample document for testing."""
        content = """
        This is the first paragraph of a test document. It contains some text that should be chunked appropriately.
        
        This is the second paragraph. It provides additional content for testing the hierarchical chunking functionality.
        The chunker should create both parent and child chunks from this content.
        
        This is the third paragraph. It extends the document further to ensure we have enough content for multiple chunks.
        The hierarchical relationship should be maintained between parent and child chunks.
        
        This is the fourth paragraph. It provides even more content for comprehensive testing of the chunking algorithm.
        We want to ensure that the token counting and overlap logic work correctly.
        """
        
        return Document(
            document_id="test_doc_1",
            content=content.strip(),
            metadata={"source": "test", "type": "sample"},
            source_path="/test/path/doc.txt"
        )
    
    def test_chunk_document_basic(self, chunker, sample_document):
        """Test basic document chunking functionality."""
        chunks = chunker.chunk_document(sample_document)
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, Chunk) for chunk in chunks)
        
        # Check that we have both parent and child chunks
        parent_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "parent"]
        child_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "child"]
        
        assert len(parent_chunks) > 0
        assert len(child_chunks) > 0
        assert len(child_chunks) >= len(parent_chunks)  # Should have at least as many children
    
    def test_chunk_document_empty_content(self, chunker):
        """Test chunking with empty document content."""
        empty_doc = Document(
            document_id="empty_doc",
            content="",
            metadata={}
        )
        
        with pytest.raises(ChunkingException) as exc_info:
            chunker.chunk_document(empty_doc)
        
        assert "Empty document" in str(exc_info.value)
    
    def test_chunk_document_whitespace_only(self, chunker):
        """Test chunking with whitespace-only content."""
        whitespace_doc = Document(
            document_id="whitespace_doc",
            content="   \n\t  \n  ",
            metadata={}
        )
        
        with pytest.raises(ChunkingException) as exc_info:
            chunker.chunk_document(whitespace_doc)
        
        assert "Empty document" in str(exc_info.value)
    
    def test_parent_child_relationships(self, chunker, sample_document):
        """Test that parent-child relationships are correctly established."""
        chunks = chunker.chunk_document(sample_document)
        
        parent_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "parent"]
        child_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "child"]
        
        # Every child should have a valid parent
        for child in child_chunks:
            assert child.parent_id is not None
            # Find the corresponding parent
            parent = next((p for p in parent_chunks if p.chunk_id == child.parent_id), None)
            assert parent is not None
            assert parent.metadata.get("chunk_type") == "parent"
    
    def test_chunk_metadata(self, chunker, sample_document):
        """Test that chunk metadata is properly set."""
        chunks = chunker.chunk_document(sample_document)
        
        for chunk in chunks:
            assert chunk.metadata["document_id"] == sample_document.document_id
            assert chunk.metadata["source_path"] == sample_document.source_path
            assert "chunk_type" in chunk.metadata
            assert "chunk_index" in chunk.metadata
            
            # Check that original document metadata is preserved
            for key, value in sample_document.metadata.items():
                assert chunk.metadata[key] == value
    
    def test_token_counting(self, chunker, sample_document):
        """Test that token counting works correctly."""
        chunks = chunker.chunk_document(sample_document)
        
        for chunk in chunks:
            assert chunk.token_count > 0
            assert isinstance(chunk.token_count, int)
            
            # Token count should roughly correlate with content length
            if len(chunk.content) > 0:
                ratio = len(chunk.content) / chunk.token_count
                assert 2 < ratio < 6  # Reasonable range for characters per token
    
    def test_chunk_batch(self, chunker):
        """Test batch chunking functionality."""
        documents = [
            Document(
                document_id=f"doc_{i}",
                content=f"This is test document {i}. " * 20,  # Enough content for chunking
                metadata={"batch_test": True, "doc_index": i}
            )
            for i in range(3)
        ]
        
        results = chunker.chunk_batch(documents)
        
        assert len(results) == 3
        for doc_id in [f"doc_{i}" for i in range(3)]:
            assert doc_id in results
            assert len(results[doc_id]) > 0
    
    def test_chunk_hierarchy_analysis(self, chunker, sample_document):
        """Test chunk hierarchy analysis functionality."""
        chunks = chunker.chunk_document(sample_document)
        hierarchy = chunker.get_chunk_hierarchy(chunks)
        
        assert "total_chunks" in hierarchy
        assert "parent_chunks" in hierarchy
        assert "child_chunks" in hierarchy
        assert "avg_parent_tokens" in hierarchy
        assert "avg_child_tokens" in hierarchy
        assert "structure" in hierarchy
        
        assert hierarchy["total_chunks"] == len(chunks)
        assert hierarchy["parent_chunks"] + hierarchy["child_chunks"] == len(chunks)
    
    def test_chunk_validation(self, chunker, sample_document):
        """Test chunk validation functionality."""
        chunks = chunker.chunk_document(sample_document)
        validation = chunker.validate_chunks(chunks)
        
        assert "valid" in validation
        assert "issues" in validation
        assert "statistics" in validation
        
        # For a properly chunked document, validation should pass
        assert validation["valid"] is True
        assert len(validation["issues"]) == 0
    
    @patch('rag_pipeline.chunking.hierarchical_chunker.tiktoken')
    def test_tokenizer_failure(self, mock_tiktoken, chunker, sample_document):
        """Test behavior when tokenizer fails to load."""
        # Mock tiktoken to raise an exception
        mock_tiktoken.get_encoding.side_effect = Exception("Tokenizer failed")
        
        # Create a new chunker to trigger tokenizer loading
        new_chunker = HierarchicalChunker()
        
        with pytest.raises(ChunkingException) as exc_info:
            new_chunker.chunk_document(sample_document)
        
        assert "Tokenizer load failed" in str(exc_info.value)
    
    def test_chunk_size_limits(self, chunker):
        """Test chunking with different size limits."""
        # Very small chunks
        small_chunker = HierarchicalChunker(
            child_chunk_size=50,
            parent_chunk_size=100,
            chunk_overlap=0.1
        )
        
        doc = Document(
            document_id="size_test",
            content="A" * 1000,  # Long content
            metadata={}
        )
        
        chunks = small_chunker.chunk_document(doc)
        
        # Should create many small chunks
        assert len(chunks) > 10
        
        # Check that chunks respect size limits (with some tolerance for overlap)
        child_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "child"]
        parent_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "parent"]
        
        for child in child_chunks:
            assert child.token_count <= small_chunker.child_chunk_size * 1.2  # 20% tolerance
        
        for parent in parent_chunks:
            assert parent.token_count <= small_chunker.parent_chunk_size * 1.2  # 20% tolerance


if __name__ == "__main__":
    pytest.main([__file__])