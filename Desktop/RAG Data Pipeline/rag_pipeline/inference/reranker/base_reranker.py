"""Abstract base class for reranker implementations."""

from abc import ABC, abstractmethod
from typing import List

from ...core.logging_config import get_logger
from ...models.reranker import RerankResult, RerankCandidate


class BaseReranker(ABC):
    """Abstract base class for all reranker implementations."""
    
    def __init__(self, model_name: str = None, **kwargs):
        self.logger = get_logger(__name__, "reranker")
        self.model_name = model_name
        self.config = kwargs
        self._initialized = False
    
    @abstractmethod
    def _initialize(self) -> None:
        """Initialize the reranker model/client."""
        pass
    
    @abstractmethod
    def _rerank_batch(
        self, 
        query: str, 
        candidates: List[RerankCandidate], 
        top_k: int
    ) -> List[RerankResult]:
        """Perform the actual reranking."""
        pass
    
    def rerank(
        self,
        query: str,
        candidates: List[RerankCandidate],
        top_k: int = 5
    ) -> List[RerankResult]:
        """Rerank candidates based on relevance to query."""
        if not candidates:
            return []
        
        if not self._initialized:
            self._initialize()
        
        # Limit candidates to reasonable number for API efficiency
        max_candidates = min(len(candidates), 1000)
        limited_candidates = candidates[:max_candidates]
        
        # Perform reranking
        results = self._rerank_batch(query, limited_candidates, top_k)
        
        self.logger.info(
            f"Reranked {len(limited_candidates)} candidates to top {len(results)}",
            extra={
                "query_length": len(query),
                "candidate_count": len(limited_candidates),
                "result_count": len(results),
                "reranker_type": self.__class__.__name__
            }
        )
        
        return results
    
    async def rerank_async(
        self,
        query: str,
        candidates: List[RerankCandidate],
        top_k: int = 5
    ) -> List[RerankResult]:
        """Asynchronously rerank candidates."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.rerank, query, candidates, top_k)
    
    def validate_candidates(self, candidates: List[RerankCandidate]) -> List[RerankCandidate]:
        """Validate and clean candidate data."""
        valid_candidates = []
        
        for candidate in candidates:
            if not candidate.content or not candidate.content.strip():
                self.logger.warning(f"Skipping empty candidate: {candidate.chunk_id}")
                continue
            
            if not candidate.chunk_id:
                self.logger.warning("Skipping candidate with missing chunk_id")
                continue
            
            valid_candidates.append(candidate)
        
        return valid_candidates
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the reranker model."""
        return {
            "model_name": self.model_name,
            "reranker_type": self.__class__.__name__,
            "initialized": self._initialized,
            "config": self.config
        }