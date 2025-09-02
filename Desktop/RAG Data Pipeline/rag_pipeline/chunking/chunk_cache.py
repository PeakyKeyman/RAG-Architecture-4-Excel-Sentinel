"""Caching layer for chunk storage and retrieval."""

import time
from typing import Any, Dict, List, Optional, Set
from datetime import datetime, timedelta, timezone
import threading
from collections import OrderedDict

from ..core.exceptions import ChunkingException
from ..core.logging_config import get_logger, log_performance
from ..models.chunk import Chunk


class ChunkCacheEntry:
    """Cache entry for chunk data."""
    
    def __init__(self, chunk: Chunk, ttl_seconds: int = 3600):
        self.chunk = chunk
        self.created_at = datetime.now(timezone.utc)
        self.expires_at = self.created_at + timedelta(seconds=ttl_seconds)
        self.access_count = 0
        self.last_accessed = self.created_at
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        return datetime.now(timezone.utc) > self.expires_at
    
    def touch(self) -> None:
        """Update access statistics."""
        self.access_count += 1
        self.last_accessed = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "chunk_id": self.chunk.chunk_id,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat()
        }


class ChunkCache:
    """In-memory cache for chunk storage with LRU eviction."""
    
    def __init__(self, max_size: int = 10000, default_ttl: int = 3600):
        self.logger = get_logger(__name__, "chunk_cache")
        self.max_size = max_size
        self.default_ttl = default_ttl
        
        self._cache: OrderedDict[str, ChunkCacheEntry] = OrderedDict()
        self._document_chunks: Dict[str, Set[str]] = {}  # document_id -> chunk_ids
        self._parent_children: Dict[str, Set[str]] = {}  # parent_id -> child_ids
        self._lock = threading.RLock()
        
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expired_removals": 0
        }
    
    def put_chunk(self, chunk: Chunk, ttl_seconds: int = None) -> None:
        """Store a chunk in the cache."""
        ttl = ttl_seconds or self.default_ttl
        
        with self._lock:
            try:
                start_time = time.time()
                
                # Create cache entry
                entry = ChunkCacheEntry(chunk, ttl)
                
                # Remove existing entry if present
                if chunk.chunk_id in self._cache:
                    self._remove_chunk_entry(chunk.chunk_id)
                
                # Add to cache
                self._cache[chunk.chunk_id] = entry
                
                # Update document mapping
                document_id = chunk.metadata.get("document_id")
                if document_id:
                    if document_id not in self._document_chunks:
                        self._document_chunks[document_id] = set()
                    self._document_chunks[document_id].add(chunk.chunk_id)
                
                # Update parent-child mapping
                if chunk.parent_id:
                    if chunk.parent_id not in self._parent_children:
                        self._parent_children[chunk.parent_id] = set()
                    self._parent_children[chunk.parent_id].add(chunk.chunk_id)
                
                # Evict if necessary
                self._evict_if_necessary()
                
                duration = (time.time() - start_time) * 1000
                log_performance(
                    self.logger,
                    "cache_put",
                    duration,
                    metadata={
                        "chunk_id": chunk.chunk_id,
                        "cache_size": len(self._cache),
                        "ttl_seconds": ttl
                    }
                )
                
            except Exception as e:
                raise ChunkingException(
                    f"Failed to cache chunk: {str(e)}",
                    component="chunk_cache",
                    error_code="CACHE_PUT_FAILED",
                    details={"chunk_id": chunk.chunk_id}
                )
    
    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        """Retrieve a chunk from the cache."""
        with self._lock:
            try:
                start_time = time.time()
                
                # Check if chunk exists
                if chunk_id not in self._cache:
                    self.stats["misses"] += 1
                    return None
                
                entry = self._cache[chunk_id]
                
                # Check if expired
                if entry.is_expired():
                    self._remove_chunk_entry(chunk_id)
                    self.stats["misses"] += 1
                    self.stats["expired_removals"] += 1
                    return None
                
                # Update access statistics
                entry.touch()
                
                # Move to end (LRU)
                self._cache.move_to_end(chunk_id)
                
                self.stats["hits"] += 1
                
                duration = (time.time() - start_time) * 1000
                log_performance(
                    self.logger,
                    "cache_get",
                    duration,
                    metadata={
                        "chunk_id": chunk_id,
                        "hit": True,
                        "access_count": entry.access_count
                    }
                )
                
                return entry.chunk
                
            except Exception as e:
                raise ChunkingException(
                    f"Failed to retrieve chunk from cache: {str(e)}",
                    component="chunk_cache",
                    error_code="CACHE_GET_FAILED",
                    details={"chunk_id": chunk_id}
                )
    
    def get_chunks_batch(self, chunk_ids: List[str]) -> Dict[str, Optional[Chunk]]:
        """Retrieve multiple chunks from the cache."""
        results = {}
        
        for chunk_id in chunk_ids:
            results[chunk_id] = self.get_chunk(chunk_id)
        
        return results
    
    def get_document_chunks(self, document_id: str) -> List[Chunk]:
        """Get all chunks for a specific document."""
        with self._lock:
            try:
                chunk_ids = self._document_chunks.get(document_id, set())
                chunks = []
                
                for chunk_id in chunk_ids:
                    chunk = self.get_chunk(chunk_id)
                    if chunk:
                        chunks.append(chunk)
                
                return chunks
                
            except Exception as e:
                raise ChunkingException(
                    f"Failed to retrieve document chunks: {str(e)}",
                    component="chunk_cache",
                    error_code="DOCUMENT_CHUNKS_FAILED",
                    details={"document_id": document_id}
                )
    
    def get_child_chunks(self, parent_id: str) -> List[Chunk]:
        """Get all child chunks for a parent chunk."""
        with self._lock:
            try:
                child_ids = self._parent_children.get(parent_id, set())
                children = []
                
                for child_id in child_ids:
                    chunk = self.get_chunk(child_id)
                    if chunk:
                        children.append(chunk)
                
                return children
                
            except Exception as e:
                raise ChunkingException(
                    f"Failed to retrieve child chunks: {str(e)}",
                    component="chunk_cache",
                    error_code="CHILD_CHUNKS_FAILED",
                    details={"parent_id": parent_id}
                )
    
    def get_parent_chunk(self, child_id: str) -> Optional[Chunk]:
        """Get the parent chunk for a child chunk."""
        child_chunk = self.get_chunk(child_id)
        if child_chunk and child_chunk.parent_id:
            return self.get_chunk(child_chunk.parent_id)
        return None
    
    def invalidate_document(self, document_id: str) -> int:
        """Remove all chunks for a document from cache."""
        with self._lock:
            try:
                chunk_ids = self._document_chunks.get(document_id, set()).copy()
                removed_count = 0
                
                for chunk_id in chunk_ids:
                    if self._remove_chunk_entry(chunk_id):
                        removed_count += 1
                
                # Clean up document mapping
                if document_id in self._document_chunks:
                    del self._document_chunks[document_id]
                
                self.logger.info(
                    f"Invalidated {removed_count} chunks for document {document_id}",
                    extra={"document_id": document_id, "removed_count": removed_count}
                )
                
                return removed_count
                
            except Exception as e:
                raise ChunkingException(
                    f"Failed to invalidate document: {str(e)}",
                    component="chunk_cache",
                    error_code="INVALIDATE_FAILED",
                    details={"document_id": document_id}
                )
    
    def clear(self) -> None:
        """Clear all entries from cache."""
        with self._lock:
            self._cache.clear()
            self._document_chunks.clear()
            self._parent_children.clear()
            self.stats = {
                "hits": 0,
                "misses": 0,
                "evictions": 0,
                "expired_removals": 0
            }
            
            self.logger.info("Cache cleared")
    
    def cleanup_expired(self) -> int:
        """Remove expired entries from cache."""
        with self._lock:
            expired_chunks = []
            
            for chunk_id, entry in self._cache.items():
                if entry.is_expired():
                    expired_chunks.append(chunk_id)
            
            removed_count = 0
            for chunk_id in expired_chunks:
                if self._remove_chunk_entry(chunk_id):
                    removed_count += 1
                    self.stats["expired_removals"] += 1
            
            if removed_count > 0:
                self.logger.info(f"Cleaned up {removed_count} expired cache entries")
            
            return removed_count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            hit_rate = self.stats["hits"] / (self.stats["hits"] + self.stats["misses"]) if (self.stats["hits"] + self.stats["misses"]) > 0 else 0
            
            return {
                **self.stats,
                "cache_size": len(self._cache),
                "max_size": self.max_size,
                "hit_rate": round(hit_rate, 3),
                "document_count": len(self._document_chunks),
                "parent_count": len(self._parent_children)
            }
    
    def _remove_chunk_entry(self, chunk_id: str) -> bool:
        """Remove a chunk entry and update all mappings."""
        if chunk_id not in self._cache:
            return False
        
        entry = self._cache[chunk_id]
        chunk = entry.chunk
        
        # Remove from main cache
        del self._cache[chunk_id]
        
        # Update document mapping
        document_id = chunk.metadata.get("document_id")
        if document_id and document_id in self._document_chunks:
            self._document_chunks[document_id].discard(chunk_id)
            if not self._document_chunks[document_id]:
                del self._document_chunks[document_id]
        
        # Update parent-child mapping
        if chunk.parent_id and chunk.parent_id in self._parent_children:
            self._parent_children[chunk.parent_id].discard(chunk_id)
            if not self._parent_children[chunk.parent_id]:
                del self._parent_children[chunk.parent_id]
        
        return True
    
    def _evict_if_necessary(self) -> None:
        """Evict oldest entries if cache is full."""
        while len(self._cache) > self.max_size:
            # Remove least recently used item
            oldest_id = next(iter(self._cache))
            self._remove_chunk_entry(oldest_id)
            self.stats["evictions"] += 1


# Global chunk cache instance
chunk_cache = ChunkCache()