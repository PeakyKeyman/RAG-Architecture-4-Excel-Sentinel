"""Hybrid search implementation combining vector similarity and keyword matching."""

import asyncio
import time
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
import re
from collections import Counter, defaultdict
import math

from ..core.config import settings
from ..core.exceptions import SearchException
from ..core.logging_config import get_logger, log_performance
from ..vector_store.vertex_vector_store import vector_store
from ..vector_store.embeddings import embedding_model
from ..models.search import VectorSearchResult, HybridSearchResult


class KeywordMatcher:
    """Keyword-based search implementation."""
    
    def __init__(self):
        self.logger = get_logger(__name__, "keyword_search")
        self.stopwords = self._load_stopwords()
    
    def _load_stopwords(self) -> Set[str]:
        """Load common English stopwords."""
        return {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'were', 'will', 'with', 'the', 'this', 'but', 'they',
            'have', 'had', 'what', 'said', 'each', 'which', 'their', 'time',
            'if', 'up', 'out', 'many', 'then', 'them', 'these', 'so', 'some',
            'her', 'would', 'make', 'like', 'into', 'him', 'only', 'two', 'more',
            'go', 'see', 'no', 'way', 'could', 'my', 'than', 'first', 'been',
            'call', 'who', 'its', 'now', 'find', 'long', 'down', 'day', 'did',
            'get', 'come', 'made', 'may', 'part'
        }
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text."""
        # Convert to lowercase and extract words
        words = re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())
        
        # Filter stopwords and short words
        keywords = [word for word in words if word not in self.stopwords and len(word) > 2]
        
        return keywords
    
    def calculate_tf_idf_score(
        self, 
        query_keywords: List[str], 
        document_text: str,
        corpus_stats: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculate TF-IDF based relevance score."""
        if not query_keywords or not document_text.strip():
            return 0.0
        
        doc_keywords = self.extract_keywords(document_text)
        doc_word_count = len(doc_keywords)
        
        if doc_word_count == 0:
            return 0.0
        
        # Calculate term frequencies
        doc_tf = Counter(doc_keywords)
        
        score = 0.0
        matched_terms = 0
        
        for query_term in query_keywords:
            if query_term in doc_tf:
                tf = doc_tf[query_term] / doc_word_count
                
                # Simple IDF approximation (would use corpus stats in production)
                idf = 1.0  # Placeholder - in production, calculate from corpus
                
                # TF-IDF score
                score += tf * idf
                matched_terms += 1
        
        # Boost score based on match coverage
        if matched_terms > 0:
            coverage_boost = matched_terms / len(query_keywords)
            score *= (1.0 + coverage_boost)
        
        return score
    
    def calculate_bm25_score(
        self,
        query_keywords: List[str],
        document_text: str,
        avg_doc_length: float = 500.0,
        k1: float = 1.5,
        b: float = 0.75
    ) -> float:
        """Calculate BM25 relevance score."""
        if not query_keywords or not document_text.strip():
            return 0.0
        
        doc_keywords = self.extract_keywords(document_text)
        doc_length = len(doc_keywords)
        
        if doc_length == 0:
            return 0.0
        
        doc_tf = Counter(doc_keywords)
        score = 0.0
        
        for term in query_keywords:
            if term in doc_tf:
                tf = doc_tf[term]
                
                # BM25 formula
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * (doc_length / avg_doc_length))
                
                score += numerator / denominator
        
        return score


class HybridSearch:
    """Hybrid search combining vector similarity and keyword matching."""
    
    def __init__(
        self,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
        min_vector_score: float = 0.1,
        min_keyword_score: float = 0.1
    ):
        self.logger = get_logger(__name__, "hybrid_search")
        
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.min_vector_score = min_vector_score
        self.min_keyword_score = min_keyword_score
        
        self.keyword_matcher = KeywordMatcher()
        
        # Ensure weights sum to 1.0
        total_weight = self.vector_weight + self.keyword_weight
        self.vector_weight /= total_weight
        self.keyword_weight /= total_weight
    
    async def search(
        self,
        query: str,
        hypothetical_docs: List[str],
        top_k: int = None,
        filters: Optional[Dict[str, Any]] = None,
        include_query_in_search: bool = True
    ) -> List[HybridSearchResult]:
        """Perform hybrid search using query and hypothetical documents."""
        top_k = top_k or settings.search_top_k
        
        if not query.strip():
            raise SearchException(
                "Empty query provided for hybrid search",
                component="hybrid_search",
                error_code="EMPTY_QUERY"
            )
        
        try:
            start_time = time.time()
            
            # Prepare search queries (original query + hypothetical docs)
            search_queries = []
            if include_query_in_search:
                search_queries.append(query)
            search_queries.extend(hypothetical_docs)
            
            # Extract keywords from original query for keyword matching
            query_keywords = self.keyword_matcher.extract_keywords(query)
            
            # Perform vector searches in parallel asynchronously
            vector_results = await self._perform_vector_searches_async(search_queries, top_k * 2, filters)
            
            # Deduplicate and combine vector results
            unique_vector_results = self._deduplicate_vector_results(vector_results)
            
            # Perform keyword scoring on vector results
            hybrid_results = self._calculate_hybrid_scores(
                unique_vector_results,
                query_keywords,
                top_k
            )
            
            duration = (time.time() - start_time) * 1000
            
            log_performance(
                self.logger,
                "hybrid_search",
                duration,
                metadata={
                    "query_length": len(query),
                    "hypothetical_docs": len(hypothetical_docs),
                    "query_keywords": len(query_keywords),
                    "vector_results": len(unique_vector_results),
                    "final_results": len(hybrid_results),
                    "top_k": top_k
                }
            )
            
            self.logger.info(
                f"Hybrid search completed: {len(hybrid_results)} results",
                extra={
                    "vector_candidates": len(unique_vector_results),
                    "final_results": len(hybrid_results),
                    "query": query[:100]
                }
            )
            
            return hybrid_results
            
        except SearchException:
            raise
        except Exception as e:
            raise SearchException(
                f"Unexpected error in hybrid search: {str(e)}",
                component="hybrid_search",
                error_code="HYBRID_SEARCH_ERROR",
                details={"query": query[:100]}
            )
    
    def _perform_vector_searches(
        self,
        queries: List[str],
        top_k: int,
        filters: Optional[Dict[str, Any]]
    ) -> List[List[VectorSearchResult]]:
        """Perform vector searches for all queries."""
        results = []
        
        for query in queries:
            try:
                query_results = vector_store.similarity_search(query, top_k, filters)
                results.append(query_results)
            except Exception as e:
                self.logger.warning(f"Vector search failed for query: {str(e)}")
                results.append([])
        
        return results
    
    async def _perform_vector_searches_async(
        self,
        queries: List[str],
        top_k: int,
        filters: Optional[Dict[str, Any]]
    ) -> List[List[VectorSearchResult]]:
        """Perform vector searches for all queries asynchronously."""
        tasks = [
            vector_store.similarity_search_async(query, top_k, filters)
            for query in queries
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions and convert results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.warning(f"Vector search failed for query {i}: {str(result)}")
                processed_results.append([])
            else:
                processed_results.append(result)
        
        return processed_results
    
    def _deduplicate_vector_results(
        self,
        vector_results_list: List[List[VectorSearchResult]]
    ) -> List[VectorSearchResult]:
        """Deduplicate vector results and keep best scores."""
        chunk_scores = {}  # chunk_id -> (result, best_score)
        
        for results in vector_results_list:
            for result in results:
                chunk_id = result.chunk_id
                
                if chunk_id not in chunk_scores or result.score > chunk_scores[chunk_id][1]:
                    chunk_scores[chunk_id] = (result, result.score)
        
        # Sort by vector score
        unique_results = [result for result, _ in chunk_scores.values()]
        unique_results.sort(key=lambda x: x.score, reverse=True)
        
        return unique_results
    
    def _calculate_hybrid_scores(
        self,
        vector_results: List[VectorSearchResult],
        query_keywords: List[str],
        top_k: int
    ) -> List[HybridSearchResult]:
        """Calculate hybrid scores using Reciprocal Rank Fusion (RRF)."""
        hybrid_results = []
        
        # Calculate keyword scores and create keyword ranking
        keyword_results = []
        for vector_result in vector_results:
            keyword_score = self.keyword_matcher.calculate_bm25_score(
                query_keywords,
                vector_result.content
            )
            keyword_results.append((vector_result, keyword_score))
        
        # Sort by keyword scores (descending)
        keyword_results.sort(key=lambda x: x[1], reverse=True)
        
        # Vector results are already sorted by vector score (descending)
        
        # Apply RRF fusion
        rrf_scores = {}  # chunk_id -> RRF score
        k = 60  # RRF constant
        
        # Add vector ranking contribution
        for rank, vector_result in enumerate(vector_results):
            chunk_id = vector_result.chunk_id
            rrf_scores[chunk_id] = 1.0 / (k + rank + 1)
        
        # Add keyword ranking contribution
        for rank, (vector_result, keyword_score) in enumerate(keyword_results):
            chunk_id = vector_result.chunk_id
            if chunk_id in rrf_scores:
                rrf_scores[chunk_id] += 1.0 / (k + rank + 1)
            else:
                rrf_scores[chunk_id] = 1.0 / (k + rank + 1)
        
        # Create hybrid results with RRF scores
        for vector_result in vector_results:
            chunk_id = vector_result.chunk_id
            rrf_score = rrf_scores.get(chunk_id, 0.0)
            
            # Find keyword score for this result
            keyword_score = 0.0
            for result, score in keyword_results:
                if result.chunk_id == chunk_id:
                    keyword_score = score
                    break
            
            # Normalize individual scores for reporting
            normalized_vector_score = min(max(vector_result.score, 0.0), 1.0)
            normalized_keyword_score = min(max(keyword_score, 0.0), 10.0) / 10.0  # BM25 can be > 1
            
            # Determine match type
            match_type = self._determine_match_type(
                normalized_vector_score,
                normalized_keyword_score
            )
            
            # Only include results that meet minimum thresholds
            if (normalized_vector_score >= self.min_vector_score or 
                normalized_keyword_score >= self.min_keyword_score):
                
                hybrid_result = HybridSearchResult(
                    chunk_id=vector_result.chunk_id,
                    parent_id=vector_result.parent_id,
                    content=vector_result.content,
                    vector_score=normalized_vector_score,
                    keyword_score=normalized_keyword_score,
                    combined_score=rrf_score,  # Use RRF score as combined score
                    match_type=match_type,
                    metadata=vector_result.metadata
                )
                hybrid_results.append(hybrid_result)
        
        # Sort by RRF combined score and return top_k
        hybrid_results.sort(key=lambda x: x.combined_score, reverse=True)
        return hybrid_results[:top_k]
    
    def _determine_match_type(
        self,
        vector_score: float,
        keyword_score: float
    ) -> str:
        """Determine the type of match based on scores."""
        vector_match = vector_score >= self.min_vector_score
        keyword_match = keyword_score >= self.min_keyword_score
        
        if vector_match and keyword_match:
            return "both"
        elif vector_match:
            return "vector"
        elif keyword_match:
            return "keyword"
        else:
            return "weak"
    
    def search_sync(
        self,
        query: str,
        hypothetical_docs: List[str],
        top_k: int = None,
        filters: Optional[Dict[str, Any]] = None,
        include_query_in_search: bool = True
    ) -> List[HybridSearchResult]:
        """Synchronous wrapper for hybrid search (backwards compatibility)."""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self.search(query, hypothetical_docs, top_k, filters, include_query_in_search)
        )
    
    def get_search_config(self) -> Dict[str, Any]:
        """Get hybrid search configuration."""
        return {
            "vector_weight": self.vector_weight,
            "keyword_weight": self.keyword_weight,
            "min_vector_score": self.min_vector_score,
            "min_keyword_score": self.min_keyword_score,
            "total_weight": self.vector_weight + self.keyword_weight
        }


# Global hybrid search instance
hybrid_search = HybridSearch()