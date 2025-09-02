"""Factory for creating reranker instances based on configuration."""

from typing import Any, Dict, Optional

from ...core.config import settings
from ...core.exceptions import RerankerException
from ...core.logging_config import get_logger
from .base_reranker import BaseReranker
from .cohere_reranker import CohereReranker


class RerankerFactory:
    """Factory class for creating reranker instances."""
    
    _rerankers = {
        "cohere": CohereReranker,
        # Future rerankers can be added here:
        # "sentence_transformers": SentenceTransformerReranker,
        # "openai": OpenAIReranker,
        # "huggingface": HuggingFaceReranker,
    }
    
    @classmethod
    def create_reranker(
        self,
        reranker_type: str = None,
        model_name: str = None,
        **kwargs
    ) -> BaseReranker:
        """Create a reranker instance based on type."""
        logger = get_logger(__name__, "reranker_factory")
        
        reranker_type = reranker_type or settings.reranker_type
        
        if reranker_type not in self._rerankers:
            available_types = list(self._rerankers.keys())
            raise RerankerException(
                f"Unknown reranker type: {reranker_type}. Available types: {available_types}",
                component="reranker_factory",
                error_code="UNKNOWN_RERANKER_TYPE",
                details={
                    "requested_type": reranker_type,
                    "available_types": available_types
                }
            )
        
        try:
            # Get reranker class
            reranker_class = self._rerankers[reranker_type]
            
            # Create instance with appropriate configuration
            if reranker_type == "cohere":
                reranker = reranker_class(
                    api_key=kwargs.get("api_key", settings.cohere_api_key),
                    model_name=model_name or "rerank-english-v3.0",
                    **kwargs
                )
            else:
                # For future reranker types
                reranker = reranker_class(
                    model_name=model_name,
                    **kwargs
                )
            
            logger.info(
                f"Created {reranker_type} reranker",
                extra={
                    "reranker_type": reranker_type,
                    "model_name": model_name,
                    "class": reranker_class.__name__
                }
            )
            
            return reranker
            
        except Exception as e:
            raise RerankerException(
                f"Failed to create {reranker_type} reranker: {str(e)}",
                component="reranker_factory",
                error_code="RERANKER_CREATION_FAILED",
                details={
                    "reranker_type": reranker_type,
                    "model_name": model_name,
                    "kwargs": kwargs
                }
            )
    
    @classmethod
    def get_available_rerankers(cls) -> Dict[str, str]:
        """Get list of available reranker types with descriptions."""
        return {
            "cohere": "Cohere Rerank API - High quality neural reranking",
            # Future descriptions:
            # "sentence_transformers": "SentenceTransformers cross-encoder models",
            # "openai": "OpenAI embedding-based reranking",
            # "huggingface": "HuggingFace transformer models",
        }
    
    @classmethod
    def register_reranker(cls, name: str, reranker_class: type) -> None:
        """Register a new reranker type."""
        if not issubclass(reranker_class, BaseReranker):
            raise RerankerException(
                f"Reranker class must inherit from BaseReranker",
                component="reranker_factory",
                error_code="INVALID_RERANKER_CLASS",
                details={"class_name": reranker_class.__name__}
            )
        
        cls._rerankers[name] = reranker_class
        logger = get_logger(__name__, "reranker_factory")
        logger.info(f"Registered new reranker type: {name}")


def create_default_reranker(**kwargs) -> BaseReranker:
    """Create a reranker using default configuration."""
    return RerankerFactory.create_reranker(**kwargs)


def switch_reranker_for_sensitive_data() -> BaseReranker:
    """Create a reranker suitable for sensitive data scenarios."""
    # This could return a self-hosted model or privacy-focused option
    # For now, return the default but with a warning
    logger = get_logger(__name__, "reranker_factory")
    logger.warning("Using default reranker for sensitive data - consider self-hosted alternative")
    
    return RerankerFactory.create_reranker(
        reranker_type=settings.reranker_type
    )