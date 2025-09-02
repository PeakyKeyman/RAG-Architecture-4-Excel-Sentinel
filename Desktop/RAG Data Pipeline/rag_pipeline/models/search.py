"""Search result data models for the RAG pipeline."""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class VectorSearchResult:
    """Container for vector search results."""
    chunk_id: str
    parent_id: str
    score: float
    content: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "parent_id": self.parent_id,
            "score": self.score,
            "content": self.content,
            "metadata": self.metadata
        }


@dataclass
class HybridSearchResult:
    """Container for hybrid search results."""
    chunk_id: str
    parent_id: str
    content: str
    vector_score: float
    keyword_score: float
    combined_score: float
    metadata: Dict[str, Any]
    match_type: str  # 'vector', 'keyword', 'both'
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "parent_id": self.parent_id,
            "content": self.content[:200] + "..." if len(self.content) > 200 else self.content,
            "vector_score": self.vector_score,
            "keyword_score": self.keyword_score,
            "combined_score": self.combined_score,
            "match_type": self.match_type,
            "metadata": self.metadata
        }