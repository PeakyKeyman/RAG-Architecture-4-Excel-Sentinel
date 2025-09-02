"""Custom exception handlers for RAG pipeline."""

from typing import Any, Dict, Optional
import traceback
from datetime import datetime


class RAGPipelineException(Exception):
    """Base exception for RAG pipeline errors."""
    
    def __init__(
        self, 
        message: str, 
        component: str = "unknown",
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.component = component
        self.error_code = error_code
        self.details = details or {}
        self.timestamp = datetime.utcnow().isoformat()
        super().__init__(message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for JSON logging."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "component": self.component,
            "error_code": self.error_code,
            "details": self.details,
            "timestamp": self.timestamp,
            "traceback": traceback.format_exc()
        }


class VectorStoreException(RAGPipelineException):
    """Exception for vector store operations."""
    pass


class ChunkingException(RAGPipelineException):
    """Exception for chunking operations."""
    pass


class EmbeddingException(RAGPipelineException):
    """Exception for embedding operations."""
    pass


class SearchException(RAGPipelineException):
    """Exception for search operations."""
    pass


class HyDEException(RAGPipelineException):
    """Exception for HyDE ensemble operations."""
    pass


class RerankerException(RAGPipelineException):
    """Exception for reranking operations."""
    pass


class InferenceException(RAGPipelineException):
    """Exception for inference pipeline operations."""
    pass


class ContextPackagingException(RAGPipelineException):
    """Exception for context packaging operations."""
    pass


class KnowledgeBaseException(RAGPipelineException):
    """Exception for knowledge base operations."""
    pass


class EvaluationException(RAGPipelineException):
    """Exception for evaluation operations."""
    pass


class APIException(RAGPipelineException):
    """Exception for API operations."""
    
    def __init__(
        self,
        message: str,
        status_code: int = 500,
        component: str = "api",
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, component, error_code, details)
        self.status_code = status_code