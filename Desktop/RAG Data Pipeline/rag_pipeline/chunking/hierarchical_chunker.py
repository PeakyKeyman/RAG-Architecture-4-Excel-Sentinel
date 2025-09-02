"""Hierarchical chunking implementation for parent-child chunk relationships."""

import re
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken

from ..core.config import settings
from ..core.exceptions import ChunkingException
from ..core.logging_config import get_logger, log_performance
from ..models.chunk import Chunk, Document


class HierarchicalChunker:
    """Hierarchical chunking engine for creating parent-child chunk relationships."""
    
    def __init__(
        self,
        child_chunk_size: int = None,
        parent_chunk_size: int = None,
        chunk_overlap: float = None
    ):
        self.logger = get_logger(__name__, "chunking")
        
        self.child_chunk_size = child_chunk_size or settings.child_chunk_size
        self.parent_chunk_size = parent_chunk_size or settings.parent_chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        
        # Initialize tokenizer for accurate token counting
        self._tokenizer = None
        self._tokenizer_name = "cl100k_base"  # GPT-4 tokenizer
        
        # Initialize text splitters
        self._parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.parent_chunk_size,
            chunk_overlap=int(self.parent_chunk_size * self.chunk_overlap),
            length_function=self._token_length,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        self._child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.child_chunk_size,
            chunk_overlap=int(self.child_chunk_size * self.chunk_overlap),
            length_function=self._token_length,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def chunk_document(self, document: Document) -> List[Chunk]:
        """Chunk a document into hierarchical parent-child chunks."""
        if not document.content.strip():
            raise ChunkingException(
                "Empty document provided for chunking",
                component="chunking",
                error_code="EMPTY_DOCUMENT",
                details={"document_id": document.document_id}
            )
        
        try:
            start_time = time.time()
            
            # Step 1: Create parent chunks
            parent_chunks = self._create_parent_chunks(document)
            
            # Step 2: Create child chunks for each parent
            all_chunks = []
            for parent_chunk in parent_chunks:
                child_chunks = self._create_child_chunks(parent_chunk, document)
                parent_chunk.children = child_chunks
                all_chunks.extend([parent_chunk] + child_chunks)
            
            duration = (time.time() - start_time) * 1000
            
            log_performance(
                self.logger,
                "chunk_document",
                duration,
                metadata={
                    "document_id": document.document_id,
                    "document_length": len(document.content),
                    "parent_chunks": len(parent_chunks),
                    "total_chunks": len(all_chunks),
                    "avg_parent_size": sum(c.token_count for c in parent_chunks) / len(parent_chunks) if parent_chunks else 0
                }
            )
            
            self.logger.info(
                f"Successfully chunked document {document.document_id}",
                extra={
                    "parent_chunks": len(parent_chunks),
                    "child_chunks": len(all_chunks) - len(parent_chunks),
                    "total_chunks": len(all_chunks)
                }
            )
            
            return all_chunks
            
        except Exception as e:
            raise ChunkingException(
                f"Failed to chunk document: {str(e)}",
                component="chunking",
                error_code="CHUNKING_FAILED",
                details={"document_id": document.document_id}
            )
    
    def chunk_batch(self, documents: List[Document]) -> Dict[str, List[Chunk]]:
        """Chunk multiple documents in batch."""
        if not documents:
            return {}
        
        try:
            start_time = time.time()
            results = {}
            failed_docs = []
            
            for document in documents:
                try:
                    chunks = self.chunk_document(document)
                    results[document.document_id] = chunks
                except ChunkingException as e:
                    failed_docs.append(document.document_id)
                    self.logger.error(
                        f"Failed to chunk document {document.document_id}: {str(e)}",
                        extra={"document_id": document.document_id}
                    )
            
            duration = (time.time() - start_time) * 1000
            
            log_performance(
                self.logger,
                "chunk_batch",
                duration,
                success=len(failed_docs) == 0,
                metadata={
                    "total_documents": len(documents),
                    "successful": len(results),
                    "failed": len(failed_docs),
                    "total_chunks": sum(len(chunks) for chunks in results.values())
                }
            )
            
            if failed_docs:
                self.logger.warning(f"Failed to chunk {len(failed_docs)} documents: {failed_docs}")
            
            return results
            
        except Exception as e:
            raise ChunkingException(
                f"Failed to chunk document batch: {str(e)}",
                component="chunking",
                error_code="BATCH_CHUNKING_FAILED",
                details={"batch_size": len(documents)}
            )
    
    def _create_parent_chunks(self, document: Document) -> List[Chunk]:
        """Create parent chunks from document."""
        text_chunks = self._parent_splitter.split_text(document.content)
        parent_chunks = []
        
        current_index = 0
        for i, chunk_text in enumerate(text_chunks):
            # Find the actual position in the original text
            start_index = document.content.find(chunk_text.strip(), current_index)
            if start_index == -1:
                start_index = current_index
            
            end_index = start_index + len(chunk_text)
            current_index = start_index + 1  # Ensure we advance past current chunk start
            
            chunk = Chunk(
                chunk_id=str(uuid.uuid4()),
                parent_id=None,
                content=chunk_text.strip(),
                start_index=start_index,
                end_index=end_index,
                token_count=self._token_length(chunk_text),
                metadata={
                    "document_id": document.document_id,
                    "chunk_type": "parent",
                    "chunk_index": i,
                    "source_path": document.source_path,
                    **document.metadata
                }
            )
            parent_chunks.append(chunk)
        
        return parent_chunks
    
    def _create_child_chunks(self, parent_chunk: Chunk, document: Document) -> List[Chunk]:
        """Create child chunks from a parent chunk."""
        text_chunks = self._child_splitter.split_text(parent_chunk.content)
        child_chunks = []
        
        current_index = 0
        for i, chunk_text in enumerate(text_chunks):
            # Calculate position relative to parent chunk
            start_index = parent_chunk.content.find(chunk_text.strip(), current_index)
            if start_index == -1:
                start_index = current_index
            
            end_index = start_index + len(chunk_text)
            current_index = start_index + 1  # Ensure we advance past current chunk start
            
            # Calculate absolute position in document
            abs_start = parent_chunk.start_index + start_index
            abs_end = parent_chunk.start_index + end_index
            
            chunk = Chunk(
                chunk_id=str(uuid.uuid4()),
                parent_id=parent_chunk.chunk_id,
                content=chunk_text.strip(),
                start_index=abs_start,
                end_index=abs_end,
                token_count=self._token_length(chunk_text),
                metadata={
                    "document_id": document.document_id,
                    "chunk_type": "child",
                    "parent_chunk_id": parent_chunk.chunk_id,
                    "chunk_index": i,
                    "source_path": document.source_path,
                    **document.metadata
                }
            )
            child_chunks.append(chunk)
        
        return child_chunks
    
    def _load_tokenizer(self) -> None:
        """Load tiktoken tokenizer for accurate token counting."""
        if self._tokenizer is None:
            try:
                self._tokenizer = tiktoken.get_encoding(self._tokenizer_name)
            except Exception as e:
                self.logger.critical(f"CRITICAL: Failed to load tiktoken tokenizer: {e}. Cannot proceed.")
                raise ChunkingException(
                    "Tokenizer load failed", 
                    component="chunking",
                    error_code="TOKENIZER_LOAD_FAILED"
                ) from e
    
    def _token_length(self, text: str) -> int:
        """Get accurate token count for text."""
        if not text.strip():
            return 0
        
        # The load tokenizer method will raise an exception if it fails.
        # No need to handle it here, let it propagate.
        if self._tokenizer is None:
            self._load_tokenizer()
        
        return len(self._tokenizer.encode(text))
    
    def get_chunk_hierarchy(self, chunks: List[Chunk]) -> Dict[str, Any]:
        """Get hierarchical structure of chunks."""
        parent_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "parent"]
        child_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "child"]
        
        hierarchy = {
            "total_chunks": len(chunks),
            "parent_chunks": len(parent_chunks),
            "child_chunks": len(child_chunks),
            "avg_parent_tokens": sum(c.token_count for c in parent_chunks) / len(parent_chunks) if parent_chunks else 0,
            "avg_child_tokens": sum(c.token_count for c in child_chunks) / len(child_chunks) if child_chunks else 0,
            "structure": {}
        }
        
        # Build parent-child mapping
        for parent in parent_chunks:
            children = [c for c in child_chunks if c.parent_id == parent.chunk_id]
            hierarchy["structure"][parent.chunk_id] = {
                "parent": parent.to_dict() if hasattr(parent, 'to_dict') else vars(parent),
                "children": [c.to_dict() if hasattr(c, 'to_dict') else vars(c) for c in children]
            }
        
        return hierarchy
    
    def validate_chunks(self, chunks: List[Chunk]) -> Dict[str, Any]:
        """Validate chunk integrity and relationships."""
        validation_results = {
            "valid": True,
            "issues": [],
            "statistics": {}
        }
        
        try:
            parent_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "parent"]
            child_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "child"]
            
            # Check parent-child relationships
            orphaned_children = []
            for child in child_chunks:
                parent_found = any(p.chunk_id == child.parent_id for p in parent_chunks)
                if not parent_found:
                    orphaned_children.append(child.chunk_id)
            
            if orphaned_children:
                validation_results["valid"] = False
                validation_results["issues"].append(f"Orphaned children: {len(orphaned_children)}")
            
            # Check token counts
            oversized_chunks = [c for c in chunks if 
                             (c.metadata.get("chunk_type") == "parent" and c.token_count > self.parent_chunk_size * 1.2) or
                             (c.metadata.get("chunk_type") == "child" and c.token_count > self.child_chunk_size * 1.2)]
            
            if oversized_chunks:
                validation_results["issues"].append(f"Oversized chunks: {len(oversized_chunks)}")
            
            # Statistics
            validation_results["statistics"] = {
                "total_chunks": len(chunks),
                "parent_chunks": len(parent_chunks),
                "child_chunks": len(child_chunks),
                "orphaned_children": len(orphaned_children),
                "oversized_chunks": len(oversized_chunks)
            }
            
            return validation_results
            
        except Exception as e:
            raise ChunkingException(
                f"Failed to validate chunks: {str(e)}",
                component="chunking",
                error_code="VALIDATION_FAILED"
            )