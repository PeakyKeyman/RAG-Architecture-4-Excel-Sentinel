"""Reranker data models for the RAG pipeline."""

from typing import Any, Dict
from dataclasses import dataclass


@dataclass
class RerankResult:
    """Container for rerank results."""
    chunk_id: str
    content: str
    score: float
    original_rank: int
    new_rank: int
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "content": self.content[:200] + "..." if len(self.content) > 200 else self.content,
            "score": self.score,
            "original_rank": self.original_rank,
            "new_rank": self.new_rank,
            "metadata": self.metadata
        }


@dataclass
class RerankCandidate:
    """Container for rerank candidate."""
    chunk_id: str
    content: str
    metadata: Dict[str, Any]
    original_score: float = 0.0