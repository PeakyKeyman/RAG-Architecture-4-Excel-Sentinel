"""Embedding model wrapper for BAAI/bge-large-en-v1.5."""

import asyncio
import time
import threading
from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

from ..core.config import settings
from ..core.exceptions import EmbeddingException
from ..core.logging_config import get_logger, log_performance


class EmbeddingModel:
    """Wrapper for BAAI/bge-large-en-v1.5 embedding model."""
    
    def __init__(self, model_name: str = None, device: str = None):
        self.logger = get_logger(__name__, "embeddings")
        self.model_name = model_name or settings.embedding_model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._lock = threading.RLock()
    
    def load(self) -> None:
        """Explicitly load the model for application startup warmup."""
        self._load_model()
        
    def _load_model(self) -> None:
        """Load the embedding model with thread safety."""
        with self._lock:
            if self._model is None:
                try:
                    start_time = time.time()
                    self._model = SentenceTransformer(self.model_name, device=self.device)
                    load_time = (time.time() - start_time) * 1000
                    
                    log_performance(
                        self.logger,
                        "model_load",
                        load_time,
                        metadata={"model": self.model_name, "device": self.device}
                    )
                    
                    self.logger.info(f"Loaded embedding model: {self.model_name} on {self.device}")
                    
                except Exception as e:
                    raise EmbeddingException(
                        f"Failed to load embedding model: {str(e)}",
                        component="embeddings",
                        error_code="MODEL_LOAD_FAILED",
                        details={"model_name": self.model_name, "device": self.device}
                    )
    
    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text string."""
        return self.embed_batch([text])[0]
    
    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Embed a batch of text strings."""
        if not texts:
            return []
            
        self._load_model()
        
        try:
            start_time = time.time()
            
            # Normalize text inputs
            processed_texts = [self._preprocess_text(text) for text in texts]
            
            # Generate embeddings
            embeddings = self._model.encode(
                processed_texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                batch_size=32
            )
            
            duration = (time.time() - start_time) * 1000
            
            log_performance(
                self.logger,
                "embed_batch",
                duration,
                metadata={
                    "batch_size": len(texts),
                    "avg_text_length": sum(len(t) for t in texts) / len(texts)
                }
            )
            
            return [emb for emb in embeddings]
            
        except Exception as e:
            raise EmbeddingException(
                f"Failed to generate embeddings: {str(e)}",
                component="embeddings",
                error_code="EMBEDDING_FAILED",
                details={"batch_size": len(texts)}
            )
    
    async def embed_batch_async(self, texts: List[str]) -> List[np.ndarray]:
        """Asynchronously embed a batch of text strings."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_batch, texts)
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text before embedding."""
        if not text or not text.strip():
            raise EmbeddingException(
                "Empty text provided for embedding",
                component="embeddings",
                error_code="EMPTY_TEXT"
            )
        
        # Add instruction prefix for better retrieval performance
        # BGE models benefit from instruction prefixes
        return f"Represent this document for retrieval: {text.strip()}"
    
    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension."""
        self._load_model()
        return self._model.get_sentence_embedding_dimension()
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        try:
            # Normalize embeddings
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Calculate cosine similarity
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            raise EmbeddingException(
                f"Failed to calculate similarity: {str(e)}",
                component="embeddings",
                error_code="SIMILARITY_FAILED"
            )


# Global embedding model instance
embedding_model = EmbeddingModel()