"""Cohere reranker implementation."""

import time
import threading
from typing import List, Optional
import requests
import json

from ...core.config import settings
from ...core.exceptions import RerankerException
from ...core.logging_config import log_performance
from .base_reranker import BaseReranker, RerankCandidate, RerankResult


class CohereReranker(BaseReranker):
    """Cohere Rerank API implementation."""
    
    def __init__(self, api_key: str = None, model_name: str = "rerank-english-v3.0", **kwargs):
        super().__init__(model_name, **kwargs)
        self.api_key = api_key or settings.cohere_api_key
        self.base_url = "https://api.cohere.ai/v1"
        self.session = None
        self._lock = threading.RLock()
        
        if not self.api_key:
            raise RerankerException(
                "Cohere API key not provided",
                component="reranker",
                error_code="MISSING_API_KEY"
            )
    
    def initialize(self) -> None:
        """Explicitly initialize the reranker for application startup."""
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize the Cohere client session with thread safety."""
        with self._lock:
            if self._initialized:
                return
            
            try:
                start_time = time.time()
                
                self.session = requests.Session()
                self.session.headers.update({
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                })
                
                # Test connection and warm up
                self._test_connection()
                
                init_time = (time.time() - start_time) * 1000
                self._initialized = True
                
                self.logger.info(
                    f"Cohere reranker initialized with model: {self.model_name}",
                    extra={"init_time_ms": init_time}
                )
                
            except Exception as e:
                raise RerankerException(
                    f"Failed to initialize Cohere reranker: {str(e)}",
                    component="reranker",
                    error_code="COHERE_INIT_FAILED",
                    details={"model": self.model_name}
                )
    
    def _test_connection(self) -> None:
        """Test the Cohere API connection."""
        try:
            response = self.session.post(
                f"{self.base_url}/rerank",
                json={
                    "model": self.model_name,
                    "query": "test query",
                    "documents": ["test document"],
                    "top_k": 1,
                    "return_documents": False
                },
                timeout=10
            )
            
            if response.status_code != 200:
                raise RerankerException(
                    f"Cohere API test failed: {response.status_code} - {response.text}",
                    component="reranker",
                    error_code="COHERE_API_ERROR"
                )
                
        except requests.RequestException as e:
            raise RerankerException(
                f"Failed to connect to Cohere API: {str(e)}",
                component="reranker",
                error_code="COHERE_CONNECTION_FAILED"
            )
    
    def _rerank_batch(
        self,
        query: str,
        candidates: List[RerankCandidate],
        top_k: int
    ) -> List[RerankResult]:
        """Perform reranking using Cohere API."""
        if not candidates:
            return []
        
        try:
            start_time = time.time()
            
            # Validate candidates
            valid_candidates = self.validate_candidates(candidates)
            if not valid_candidates:
                return []
            
            # Prepare documents for Cohere API
            documents = [candidate.content for candidate in valid_candidates]
            
            # Make API request
            response = self.session.post(
                f"{self.base_url}/rerank",
                json={
                    "model": self.model_name,
                    "query": query,
                    "documents": documents,
                    "top_k": min(top_k, len(documents)),
                    "return_documents": False  # We already have the content
                },
                timeout=30
            )
            
            if response.status_code != 200:
                raise RerankerException(
                    f"Cohere rerank failed: {response.status_code} - {response.text}",
                    component="reranker",
                    error_code="COHERE_RERANK_FAILED",
                    details={"status_code": response.status_code}
                )
            
            # Parse response
            result_data = response.json()
            results = []
            
            for i, result in enumerate(result_data.get("results", [])):
                original_index = result["index"]
                candidate = valid_candidates[original_index]
                
                rerank_result = RerankResult(
                    chunk_id=candidate.chunk_id,
                    content=candidate.content,
                    score=float(result["relevance_score"]),
                    original_rank=original_index,
                    new_rank=i,
                    metadata={
                        **candidate.metadata,
                        "reranker": "cohere",
                        "model": self.model_name,
                        "original_score": candidate.original_score
                    }
                )
                results.append(rerank_result)
            
            duration = (time.time() - start_time) * 1000
            
            log_performance(
                self.logger,
                "cohere_rerank",
                duration,
                metadata={
                    "query_length": len(query),
                    "candidate_count": len(valid_candidates),
                    "result_count": len(results),
                    "top_k": top_k,
                    "model": self.model_name
                }
            )
            
            return results
            
        except RerankerException:
            raise
        except Exception as e:
            raise RerankerException(
                f"Unexpected error in Cohere reranking: {str(e)}",
                component="reranker",
                error_code="COHERE_UNEXPECTED_ERROR",
                details={
                    "query": query[:100],
                    "candidate_count": len(candidates)
                }
            )
    
    def get_model_info(self) -> dict:
        """Get Cohere model information."""
        info = super().get_model_info()
        info.update({
            "api_base": self.base_url,
            "provider": "cohere",
            "has_api_key": bool(self.api_key)
        })
        return info