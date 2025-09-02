"""Core chunk data models for the RAG pipeline."""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class Chunk:
    """Container for chunk data."""
    chunk_id: str
    parent_id: Optional[str]
    content: str
    start_index: int
    end_index: int
    token_count: int
    metadata: Dict[str, Any]
    children: Optional[List['Chunk']] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "parent_id": self.parent_id,
            "content": self.content,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "token_count": self.token_count,
            "metadata": self.metadata,
            "children": [child.to_dict() for child in self.children] if self.children else None
        }


@dataclass
class Document:
    """Container for document data."""
    document_id: str
    content: str
    metadata: Dict[str, Any]
    source_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary."""
        return {
            "document_id": self.document_id,
            "content": self.content,
            "metadata": self.metadata,
            "source_path": self.source_path
        }