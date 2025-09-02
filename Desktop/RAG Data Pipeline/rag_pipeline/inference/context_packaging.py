"""Context packaging for LLM consumption with hierarchical chunk retrieval."""

import time
import tiktoken
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

from ..core.config import settings
from ..core.exceptions import ContextPackagingException
from ..core.logging_config import get_logger, log_performance
from ..chunking.chunk_cache import chunk_cache
from ..models.chunk import Chunk
from ..models.reranker import RerankResult


@dataclass
class PackagedContext:
    """Container for packaged context ready for LLM consumption."""
    contexts: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    total_tokens: int
    context_structure: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "contexts": self.contexts,
            "metadata": self.metadata,
            "total_tokens": self.total_tokens,
            "context_structure": self.context_structure
        }
    
    def format_for_prompt(self, include_metadata: bool = True) -> str:
        """Format context for inclusion in LLM prompt."""
        formatted_parts = []
        
        for i, context in enumerate(self.contexts, 1):
            content_part = f"Context {i}:\n{context['content']}\n"
            
            if include_metadata and context.get('source_info'):
                content_part += f"Source: {context['source_info']}\n"
            
            formatted_parts.append(content_part)
        
        return "\n".join(formatted_parts)


class ContextPackager:
    """Context packaging engine for hierarchical chunk organization."""
    
    def __init__(self, max_context_tokens: int = 4000):
        self.logger = get_logger(__name__, "context_packaging")
        self.max_context_tokens = max_context_tokens
        self._tokenizer = None
        self._tokenizer_name = "cl100k_base"  # GPT-4 tokenizer
    
    def package_context(
        self,
        reranked_results: List[RerankResult],
        query: str,
        include_parent_chunks: bool = True,
        deduplicate_parents: bool = True,
        context_window_tokens: int = None
    ) -> PackagedContext:
        """Package reranked results into structured context for LLM."""
        if not reranked_results:
            return PackagedContext(
                contexts=[],
                metadata={"query": query, "result_count": 0},
                total_tokens=0,
                context_structure="empty"
            )
        
        context_window = context_window_tokens or self.max_context_tokens
        
        try:
            start_time = time.time()
            
            # Retrieve parent chunks if requested
            if include_parent_chunks:
                enhanced_results = self._retrieve_parent_chunks(reranked_results)
            else:
                enhanced_results = [(result, None) for result in reranked_results]
            
            # Build context hierarchy
            context_hierarchy = self._build_context_hierarchy(enhanced_results)
            
            # Package contexts with token management
            packaged_context = self._package_with_token_limit(
                context_hierarchy,
                query,
                context_window,
                deduplicate_parents
            )
            
            duration = (time.time() - start_time) * 1000
            
            log_performance(
                self.logger,
                "package_context",
                duration,
                metadata={
                    "query_length": len(query),
                    "input_results": len(reranked_results),
                    "final_contexts": len(packaged_context.contexts),
                    "total_tokens": packaged_context.total_tokens,
                    "include_parents": include_parent_chunks,
                    "context_structure": packaged_context.context_structure
                }
            )
            
            self.logger.info(
                f"Packaged context: {len(packaged_context.contexts)} contexts, "
                f"{packaged_context.total_tokens} tokens",
                extra={
                    "context_count": len(packaged_context.contexts),
                    "total_tokens": packaged_context.total_tokens,
                    "structure": packaged_context.context_structure
                }
            )
            
            return packaged_context
            
        except Exception as e:
            raise ContextPackagingException(
                f"Failed to package context: {str(e)}",
                component="context_packaging",
                error_code="PACKAGING_FAILED",
                details={
                    "query": query[:100],
                    "result_count": len(reranked_results)
                }
            )
    
    def _retrieve_parent_chunks(
        self,
        reranked_results: List[RerankResult]
    ) -> List[Tuple[RerankResult, Optional[Chunk]]]:
        """Retrieve parent chunks for child chunks."""
        enhanced_results = []
        cache_hits = 0
        cache_misses = 0
        
        for result in reranked_results:
            parent_chunk = None
            
            # Check if this is a child chunk (has metadata indicating parent)
            parent_id = result.metadata.get("parent_chunk_id")
            if parent_id:
                # Try to get parent from cache
                parent_chunk = chunk_cache.get_chunk(parent_id)
                
                if parent_chunk:
                    cache_hits += 1
                else:
                    cache_misses += 1
                    self.logger.warning(
                        f"Parent chunk not found in cache: {parent_id}",
                        extra={"child_chunk_id": result.chunk_id, "parent_id": parent_id}
                    )
            
            enhanced_results.append((result, parent_chunk))
        
        self.logger.debug(
            f"Parent chunk retrieval: {cache_hits} hits, {cache_misses} misses",
            extra={"cache_hits": cache_hits, "cache_misses": cache_misses}
        )
        
        return enhanced_results
    
    def _build_context_hierarchy(
        self,
        enhanced_results: List[Tuple[RerankResult, Optional[Chunk]]]
    ) -> List[Dict[str, Any]]:
        """Build hierarchical context structure."""
        context_hierarchy = []
        
        for result, parent_chunk in enhanced_results:
            # Determine which content to use
            if parent_chunk and parent_chunk.content.strip():
                # Use parent chunk content for better context
                primary_content = parent_chunk.content
                content_type = "parent"
                chunk_info = {
                    "child_chunk_id": result.chunk_id,
                    "parent_chunk_id": parent_chunk.chunk_id,
                    "child_score": result.score
                }
            else:
                # Use child chunk content
                primary_content = result.content
                content_type = "child"
                chunk_info = {
                    "chunk_id": result.chunk_id,
                    "score": result.score
                }
            
            # Calculate token count
            token_count = self._estimate_tokens(primary_content)
            
            context_item = {
                "content": primary_content,
                "content_type": content_type,
                "token_count": token_count,
                "relevance_score": result.score,
                "original_rank": result.original_rank,
                "new_rank": result.new_rank,
                "chunk_info": chunk_info,
                "source_info": self._extract_source_info(result.metadata),
                "metadata": result.metadata
            }
            
            context_hierarchy.append(context_item)
        
        return context_hierarchy
    
    def _package_with_token_limit(
        self,
        context_hierarchy: List[Dict[str, Any]],
        query: str,
        max_tokens: int,
        deduplicate_parents: bool
    ) -> PackagedContext:
        """Package contexts while respecting token limits."""
        selected_contexts = []
        total_tokens = 0
        used_parent_ids = set()
        
        # Reserve tokens for query and formatting
        reserved_tokens = len(query.split()) + 100  # Rough estimate
        available_tokens = max_tokens - reserved_tokens
        
        for context_item in context_hierarchy:
            item_tokens = context_item["token_count"]
            
            # Check for parent deduplication
            if deduplicate_parents and context_item["content_type"] == "parent":
                parent_id = context_item["chunk_info"].get("parent_chunk_id")
                if parent_id in used_parent_ids:
                    continue
                used_parent_ids.add(parent_id)
            
            # Check if we can fit this context
            if total_tokens + item_tokens <= available_tokens:
                selected_contexts.append(context_item)
                total_tokens += item_tokens
            else:
                # Attempt to truncate and fit any oversized chunk
                remaining_space = available_tokens - total_tokens
                if remaining_space > 100:  # Set a minimum threshold for truncation
                    truncated_content = self._truncate_content(
                        context_item["content"],
                        remaining_space
                    )
                    if truncated_content:
                        truncated_item = context_item.copy()
                        truncated_item["content"] = truncated_content
                        truncated_item["token_count"] = self._estimate_tokens(truncated_content)
                        truncated_item["truncated"] = True
                        
                        selected_contexts.append(truncated_item)
                        total_tokens += truncated_item["token_count"]
                        
                        # Break after truncating since we've filled remaining space
                        break
                # If we can't truncate or it's too small, continue to next items
        
        # Determine context structure
        structure = self._analyze_context_structure(selected_contexts)
        
        # Create metadata
        metadata = {
            "query": query,
            "result_count": len(selected_contexts),
            "total_input_contexts": len(context_hierarchy),
            "token_limit": max_tokens,
            "reserved_tokens": reserved_tokens,
            "available_tokens": available_tokens,
            "parent_chunks": sum(1 for c in selected_contexts if c["content_type"] == "parent"),
            "child_chunks": sum(1 for c in selected_contexts if c["content_type"] == "child"),
            "truncated_contexts": sum(1 for c in selected_contexts if c.get("truncated", False)),
            "deduplication_enabled": deduplicate_parents
        }
        
        return PackagedContext(
            contexts=selected_contexts,
            metadata=metadata,
            total_tokens=total_tokens,
            context_structure=structure
        )
    
    def _load_tokenizer(self) -> None:
        """Load tiktoken tokenizer for accurate token counting."""
        if self._tokenizer is None:
            try:
                self._tokenizer = tiktoken.get_encoding(self._tokenizer_name)
            except Exception as e:
                self.logger.critical(f"CRITICAL: Failed to load tiktoken tokenizer: {e}. Cannot proceed.")
                raise ContextPackagingException(
                    "Tokenizer load failed",
                    component="context_packaging", 
                    error_code="TOKENIZER_LOAD_FAILED"
                ) from e
    
    def _estimate_tokens(self, text: str) -> int:
        """Get accurate token count for text."""
        if not text.strip():
            return 0
        
        # Use tiktoken tokenizer for accuracy
        if self._tokenizer is None:
            self._load_tokenizer()
        
        return len(self._tokenizer.encode(text))
    
    def _truncate_content(self, content: str, max_tokens: int) -> Optional[str]:
        """Truncate content to fit within token limit."""
        if max_tokens <= 0:
            return None
        
        # Estimate character limit
        max_chars = max_tokens * 4
        
        if len(content) <= max_chars:
            return content
        
        # Truncate at sentence boundary if possible
        truncated = content[:max_chars]
        
        # Find last sentence ending
        sentence_endings = ['. ', '! ', '? ', '.\n', '!\n', '?\n']
        last_sentence_end = -1
        
        for ending in sentence_endings:
            pos = truncated.rfind(ending)
            if pos > last_sentence_end:
                last_sentence_end = pos + len(ending) - 1
        
        if last_sentence_end > max_chars // 2:  # Only use if we keep at least half
            return truncated[:last_sentence_end + 1]
        else:
            return truncated + "..."
    
    def _extract_source_info(self, metadata: Dict[str, Any]) -> str:
        """Extract readable source information from metadata."""
        parts = []
        
        if "document_id" in metadata:
            parts.append(f"Document: {metadata['document_id']}")
        
        if "source_path" in metadata:
            parts.append(f"File: {metadata['source_path']}")
        
        if "chunk_index" in metadata:
            parts.append(f"Section: {metadata['chunk_index']}")
        
        return " | ".join(parts) if parts else "Unknown source"
    
    def _analyze_context_structure(self, contexts: List[Dict[str, Any]]) -> str:
        """Analyze the structure of the packaged contexts."""
        if not contexts:
            return "empty"
        
        parent_count = sum(1 for c in contexts if c["content_type"] == "parent")
        child_count = sum(1 for c in contexts if c["content_type"] == "child")
        
        if parent_count > 0 and child_count == 0:
            return "parent_only"
        elif parent_count == 0 and child_count > 0:
            return "child_only"
        elif parent_count > 0 and child_count > 0:
            return "mixed_hierarchy"
        else:
            return "unknown"
    
    def optimize_context_order(
        self,
        packaged_context: PackagedContext,
        strategy: str = "relevance"
    ) -> PackagedContext:
        """Optimize the order of contexts for better LLM performance."""
        if strategy == "relevance":
            # Sort by relevance score (already done by reranker)
            pass
        elif strategy == "chronological":
            # Sort by document order if available
            packaged_context.contexts.sort(
                key=lambda x: x["metadata"].get("chunk_index", 0)
            )
        elif strategy == "similarity_clustering":
            # Group similar contexts together (would need similarity calculation)
            pass
        
        # Update metadata
        packaged_context.metadata["context_order_strategy"] = strategy
        
        return packaged_context


# Global context packager instance
context_packager = ContextPackager()