"""Configuration management for RAG pipeline."""

import os
from typing import Optional
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # API Configuration
    api_key: str = Field(..., env="RAG_API_KEY")
    api_host: str = Field(default="0.0.0.0", env="RAG_API_HOST")
    api_port: int = Field(default=8000, env="RAG_API_PORT")
    
    # Google Vertex AI Configuration
    gcp_project_id: str = Field(..., env="GCP_PROJECT_ID")
    gcp_location: str = Field(default="us-central1", env="GCP_LOCATION")
    vertex_vector_index: str = Field(..., env="VERTEX_VECTOR_INDEX")
    vertex_endpoint: str = Field(..., env="VERTEX_ENDPOINT")
    
    # Embedding Model Configuration
    embedding_model: str = Field(default="BAAI/bge-large-en-v1.5", env="EMBEDDING_MODEL")
    embedding_dimension: int = Field(default=1024, env="EMBEDDING_DIMENSION")
    
    # Chunking Configuration
    child_chunk_size: int = Field(default=400, env="CHILD_CHUNK_SIZE")
    parent_chunk_size: int = Field(default=1200, env="PARENT_CHUNK_SIZE")
    chunk_overlap: float = Field(default=0.15, env="CHUNK_OVERLAP")
    
    # Search Configuration
    search_top_k: int = Field(default=50, env="SEARCH_TOP_K")
    rerank_top_k: int = Field(default=5, env="RERANK_TOP_K")
    
    # HyDE Configuration
    hyde_ensemble_size: int = Field(default=3, env="HYDE_ENSEMBLE_SIZE")
    
    # Reranker Configuration
    reranker_type: str = Field(default="cohere", env="RERANKER_TYPE")
    cohere_api_key: Optional[str] = Field(default=None, env="COHERE_API_KEY")
    
    # Gemini Configuration
    gemini_model: str = Field(default="gemini-1.5-pro", env="GEMINI_MODEL")
    gemini_api_key: str = Field(..., env="GEMINI_API_KEY")
    gemini_temperature: float = Field(default=0.1, env="GEMINI_TEMPERATURE")
    
    # Performance Configuration
    max_concurrent_requests: int = Field(default=10, env="MAX_CONCURRENT_REQUESTS")
    request_timeout: int = Field(default=30, env="REQUEST_TIMEOUT")
    
    # Logging Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: Optional[str] = Field(default=None, env="LOG_FILE")
    
    # Evaluation Configuration
    langsmith_api_key: Optional[str] = Field(default=None, env="LANGSMITH_API_KEY")
    langsmith_project: str = Field(default="rag-pipeline", env="LANGSMITH_PROJECT")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()