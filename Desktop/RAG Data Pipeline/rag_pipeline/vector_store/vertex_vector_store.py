"""Google Vertex AI Vector Search interface."""

import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple
import uuid
from google.cloud import aiplatform
from google.cloud.aiplatform import MatchingEngineIndex, MatchingEngineIndexEndpoint

from ..core.config import settings
from ..core.exceptions import VectorStoreException
from ..core.logging_config import get_logger, log_performance
from ..models.search import VectorSearchResult
from .embeddings import embedding_model


class VertexVectorStore:
    """Google Vertex AI Vector Search interface."""
    
    def __init__(self):
        self.logger = get_logger(__name__, "vector_store")
        self.project_id = settings.gcp_project_id
        self.location = settings.gcp_location
        self.index_name = settings.vertex_vector_index
        self.endpoint_name = settings.vertex_endpoint
        
        self._index = None
        self._endpoint = None
        self._initialized = False
    
    def _initialize(self) -> None:
        """Initialize Vertex AI Vector Search components."""
        if self._initialized:
            return
            
        try:
            start_time = time.time()
            
            # Initialize AI Platform
            aiplatform.init(
                project=self.project_id,
                location=self.location
            )
            
            # Get index reference
            self._index = MatchingEngineIndex(self.index_name)
            
            # Get endpoint reference  
            self._endpoint = MatchingEngineIndexEndpoint(self.endpoint_name)
            
            self._initialized = True
            
            init_time = (time.time() - start_time) * 1000
            log_performance(
                self.logger,
                "vertex_initialization",
                init_time,
                metadata={
                    "project": self.project_id,
                    "location": self.location,
                    "index": self.index_name
                }
            )
            
            self.logger.info("Vertex AI Vector Search initialized successfully")
            
        except Exception as e:
            raise VectorStoreException(
                f"Failed to initialize Vertex AI Vector Search: {str(e)}",
                component="vector_store",
                error_code="VERTEX_INIT_FAILED",
                details={
                    "project": self.project_id,
                    "location": self.location,
                    "index": self.index_name
                }
            )
    
    def upsert_chunks(
        self, 
        chunks: List[Dict[str, Any]], 
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """Upsert chunks into the vector store."""
        self._initialize()
        
        if not chunks:
            return {"upserted_count": 0, "failed_count": 0}
        
        try:
            start_time = time.time()
            
            # Process in batches
            upserted_count = 0
            failed_count = 0
            
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                
                try:
                    self._upsert_batch(batch)
                    upserted_count += len(batch)
                    
                except Exception as e:
                    failed_count += len(batch)
                    self.logger.error(
                        f"Failed to upsert batch {i//batch_size + 1}: {str(e)}",
                        extra={"batch_start": i, "batch_size": len(batch)}
                    )
            
            duration = (time.time() - start_time) * 1000
            
            log_performance(
                self.logger,
                "upsert_chunks",
                duration,
                success=failed_count == 0,
                metadata={
                    "total_chunks": len(chunks),
                    "upserted_count": upserted_count,
                    "failed_count": failed_count,
                    "batch_size": batch_size
                }
            )
            
            return {
                "upserted_count": upserted_count,
                "failed_count": failed_count
            }
            
        except Exception as e:
            raise VectorStoreException(
                f"Failed to upsert chunks: {str(e)}",
                component="vector_store",
                error_code="UPSERT_FAILED",
                details={"chunk_count": len(chunks)}
            )
    
    def _upsert_batch(self, chunks: List[Dict[str, Any]]) -> None:
        """Upsert a single batch of chunks."""
        # Extract texts for embedding
        texts = [chunk.get("content", "") for chunk in chunks]
        
        # Generate embeddings
        embeddings = embedding_model.embed_batch(texts)
        
        # Prepare datapoints for Vertex AI
        datapoints = []
        for i, chunk in enumerate(chunks):
            chunk_id = chunk.get("chunk_id") or str(uuid.uuid4())
            
            datapoint = {
                "datapoint_id": chunk_id,
                "feature_vector": embeddings[i].tolist(),
                "restricts": [
                    {"namespace": "parent_id", "allow": [chunk.get("parent_id", "")]},
                    {"namespace": "document_id", "allow": [chunk.get("document_id", "")]},
                ],
                "numeric_restricts": []
            }
            
            # Add metadata as restricts with proper type handling
            metadata = chunk.get("metadata", {})
            for key, value in metadata.items():
                if isinstance(value, str):
                    datapoint["restricts"].append({
                        "namespace": key,
                        "allow": [value]
                    })
                elif isinstance(value, (int, float)):
                    # Use correct parameter for numeric values
                    datapoint["numeric_restricts"].append({
                        "namespace": key,
                        "value_float": float(value)
                    })
                elif isinstance(value, bool):
                    datapoint["restricts"].append({
                        "namespace": f"{key}_bool",
                        "allow": [str(value).lower()]
                    })
            
            datapoints.append(datapoint)
        
        # Upsert to Vertex AI
        self._index.upsert_datapoints(datapoints=datapoints)
    
    def similarity_search(
        self,
        query: str,
        top_k: int = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """Perform similarity search."""
        self._initialize()
        
        top_k = top_k or settings.search_top_k
        
        try:
            start_time = time.time()
            
            # Generate query embedding
            query_embedding = embedding_model.embed_single(query)
            
            # Prepare filters
            restricts = []
            if filters:
                for key, value in filters.items():
                    if isinstance(value, list):
                        restricts.append({"namespace": key, "allow": value})
                    else:
                        restricts.append({"namespace": key, "allow": [str(value)]})
            
            # Perform search
            response = self._endpoint.find_neighbors(
                deployed_index_id=self.index_name,
                queries=[query_embedding.tolist()],
                num_neighbors=top_k,
                restricts=restricts if restricts else None
            )
            
            # Process results
            results = []
            if response and len(response) > 0:
                neighbors = response[0]
                
                for neighbor in neighbors:
                    # Extract metadata from restricts
                    metadata = {}
                    parent_id = ""
                    
                    if hasattr(neighbor, 'restricts'):
                        for restrict in neighbor.restricts:
                            namespace = restrict.get('namespace', '')
                            allow_values = restrict.get('allow', [])
                            
                            if namespace == 'parent_id' and allow_values:
                                parent_id = allow_values[0]
                            elif namespace != 'parent_id' and allow_values:
                                metadata[namespace] = allow_values[0]
                    
                    result = VectorSearchResult(
                        chunk_id=neighbor.datapoint_id,
                        parent_id=parent_id,
                        score=1.0 - float(neighbor.distance),  # Convert distance to similarity
                        content="",  # Content retrieval would need separate storage
                        metadata=metadata
                    )
                    results.append(result)
            
            duration = (time.time() - start_time) * 1000
            
            log_performance(
                self.logger,
                "similarity_search",
                duration,
                metadata={
                    "query_length": len(query),
                    "top_k": top_k,
                    "results_count": len(results),
                    "has_filters": bool(filters)
                }
            )
            
            return results
            
        except Exception as e:
            raise VectorStoreException(
                f"Failed to perform similarity search: {str(e)}",
                component="vector_store",
                error_code="SEARCH_FAILED",
                details={"query": query[:100], "top_k": top_k}
            )
    
    async def similarity_search_async(
        self,
        query: str,
        top_k: int = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """Asynchronously perform similarity search."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.similarity_search, query, top_k, filters)
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector index."""
        self._initialize()
        
        try:
            # This would need to be implemented based on Vertex AI's actual stats API
            # Placeholder for now
            return {
                "total_vectors": 0,
                "dimension": embedding_model.get_embedding_dimension(),
                "index_name": self.index_name,
                "status": "active"
            }
            
        except Exception as e:
            raise VectorStoreException(
                f"Failed to get index stats: {str(e)}",
                component="vector_store",
                error_code="STATS_FAILED"
            )


# Global vector store instance
vector_store = VertexVectorStore()