"""
Multi-tenant vector store with org-specific indexes and security filtering.
Manages separate Vertex AI Vector Search indexes per organization for data isolation.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple
import hashlib
from google.cloud import aiplatform
from google.cloud.aiplatform import MatchingEngineIndex, MatchingEngineIndexEndpoint
from google.api_core import exceptions as gcp_exceptions

from ..core.config import settings
from ..core.exceptions import VectorStoreException
from ..core.logging_config import get_logger
from ..models.search import VectorSearchResult
from .embeddings import embedding_model


logger = get_logger(__name__, "multi_tenant_vector_store")


class OrgVectorIndex:
    """Manages a single organization's vector index."""
    
    def __init__(self, org_id: str, project_id: str, location: str):
        self.org_id = org_id
        self.project_id = project_id
        self.location = location
        
        # Generate org-specific index names
        self.index_name = f"rag-pipeline-{org_id}"
        self.endpoint_name = f"rag-pipeline-endpoint-{org_id}"
        self.deployed_index_id = f"rag-pipeline-deployed-{org_id}"
        
        self._index = None
        self._endpoint = None
        self._initialized = False
        self.logger = logger.bind(org_id=org_id)
    
    async def initialize(self) -> bool:
        """Initialize or create the org-specific vector index."""
        if self._initialized:
            return True
            
        try:
            # Initialize AI Platform for this org
            aiplatform.init(project=self.project_id, location=self.location)
            
            # Try to get existing index
            try:
                indexes = aiplatform.MatchingEngineIndex.list(
                    filter=f'display_name="{self.index_name}"'
                )
                if indexes:
                    self._index = indexes[0]
                    self.logger.info(f"Found existing index for org {self.org_id}")
                else:
                    self._index = await self._create_index()
                    
            except Exception as e:
                self.logger.warning(f"Could not find existing index, creating new one: {str(e)}")
                self._index = await self._create_index()
            
            # Try to get existing endpoint
            try:
                endpoints = aiplatform.MatchingEngineIndexEndpoint.list(
                    filter=f'display_name="{self.endpoint_name}"'
                )
                if endpoints:
                    self._endpoint = endpoints[0]
                    self.logger.info(f"Found existing endpoint for org {self.org_id}")
                else:
                    self._endpoint = await self._create_endpoint()
                    
            except Exception as e:
                self.logger.warning(f"Could not find existing endpoint, creating new one: {str(e)}")
                self._endpoint = await self._create_endpoint()
            
            # Ensure index is deployed to endpoint
            await self._ensure_index_deployed()
            
            self._initialized = True
            self.logger.info(f"Successfully initialized vector index for org {self.org_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize vector index for org {self.org_id}: {str(e)}", 
                             exc_info=True)
            return False
    
    async def _create_index(self) -> MatchingEngineIndex:
        """Create a new vector index for the organization."""
        self.logger.info(f"Creating new vector index for org {self.org_id}")
        
        index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
            display_name=self.index_name,
            dimensions=1024,  # BGE-large embedding dimension
            approximate_neighbors_count=100,
            distance_measure_type="DOT_PRODUCT_DISTANCE",
            leaf_node_embedding_count=500,
            leaf_nodes_to_search_percent=10,
            description=f"RAG pipeline vector index for organization {self.org_id}",
            labels={"org_id": self.org_id, "purpose": "rag_pipeline"}
        )
        
        self.logger.info(f"Created vector index {index.resource_name} for org {self.org_id}")
        return index
    
    async def _create_endpoint(self) -> MatchingEngineIndexEndpoint:
        """Create a new index endpoint for the organization."""
        self.logger.info(f"Creating new index endpoint for org {self.org_id}")
        
        endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
            display_name=self.endpoint_name,
            description=f"RAG pipeline index endpoint for organization {self.org_id}",
            labels={"org_id": self.org_id, "purpose": "rag_pipeline"},
            public_endpoint_enabled=True
        )
        
        self.logger.info(f"Created index endpoint {endpoint.resource_name} for org {self.org_id}")
        return endpoint
    
    async def _ensure_index_deployed(self) -> None:
        """Ensure the index is deployed to the endpoint."""
        if not self._index or not self._endpoint:
            raise VectorStoreException("Index or endpoint not initialized")
        
        try:
            # Check if index is already deployed
            deployed_indexes = self._endpoint.deployed_indexes
            
            for deployed_index in deployed_indexes:
                if deployed_index.index == self._index.resource_name:
                    self.logger.info(f"Index already deployed for org {self.org_id}")
                    return
            
            # Deploy the index
            self.logger.info(f"Deploying index to endpoint for org {self.org_id}")
            
            self._endpoint.deploy_index(
                index=self._index,
                deployed_index_id=self.deployed_index_id,
                display_name=f"Deployed RAG Index {self.org_id}",
                machine_type="n1-standard-2",  # Adjust based on needs
                min_replica_count=1,
                max_replica_count=1
            )
            
            self.logger.info(f"Successfully deployed index for org {self.org_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to deploy index for org {self.org_id}: {str(e)}")
            raise
    
    async def upsert_vectors(self, vectors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Upsert vectors to the org-specific index."""
        if not self._initialized:
            raise VectorStoreException(f"Vector store not initialized for org {self.org_id}")
        
        try:
            # Prepare vectors for Vertex AI format
            datapoints = []
            for vector_data in vectors:
                # Ensure security metadata is present
                if not all(k in vector_data.get('metadata', {}) for k in ['org_id', 'user_id']):
                    raise VectorStoreException("Missing required security metadata")
                
                if vector_data['metadata']['org_id'] != self.org_id:
                    raise VectorStoreException(f"Vector org_id mismatch: {vector_data['metadata']['org_id']} != {self.org_id}")
                
                # Generate embedding if not provided
                if 'vector' not in vector_data:
                    content = vector_data.get('content', '')
                    if not content:
                        continue
                    vector_data['vector'] = await embedding_model.embed_query_async(content)
                
                datapoint = {
                    'datapoint_id': vector_data['chunk_id'],
                    'feature_vector': vector_data['vector'],
                    'restricts': [
                        {'namespace': 'org_id', 'allow': [self.org_id]},
                        {'namespace': 'user_id', 'allow': [vector_data['metadata']['user_id']]},
                        {'namespace': 'group_id', 'allow': [vector_data['metadata'].get('group_id', 'default')]}
                    ]
                }
                datapoints.append(datapoint)
            
            if not datapoints:
                return {"upserted_count": 0, "failed_count": len(vectors)}
            
            # Upsert to index
            self._index.upsert_datapoints(datapoints=datapoints)
            
            self.logger.info(f"Upserted {len(datapoints)} vectors for org {self.org_id}")
            
            return {
                "upserted_count": len(datapoints),
                "failed_count": len(vectors) - len(datapoints)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to upsert vectors for org {self.org_id}: {str(e)}", 
                             exc_info=True)
            raise VectorStoreException(f"Vector upsert failed: {str(e)}")
    
    async def similarity_search(self, 
                               query: str,
                               top_k: int = 10,
                               user_id: str = None,
                               group_ids: List[str] = None,
                               filters: Dict[str, Any] = None) -> List[VectorSearchResult]:
        """Perform similarity search with security filtering."""
        if not self._initialized:
            raise VectorStoreException(f"Vector store not initialized for org {self.org_id}")
        
        try:
            # Generate query embedding
            query_vector = await embedding_model.embed_query_async(query)
            
            # Build security restrictions
            restricts = [{'namespace': 'org_id', 'allow': [self.org_id]}]
            
            if user_id:
                restricts.append({'namespace': 'user_id', 'allow': [user_id]})
            
            if group_ids:
                restricts.append({'namespace': 'group_id', 'allow': group_ids})
            
            # Perform search
            response = self._endpoint.find_neighbors(
                deployed_index_id=self.deployed_index_id,
                queries=[query_vector],
                num_neighbors=top_k,
                restricts=restricts
            )
            
            # Convert to VectorSearchResult objects
            results = []
            if response and len(response) > 0:
                for neighbor in response[0]:
                    results.append(VectorSearchResult(
                        chunk_id=neighbor.id,
                        content="",  # Content needs to be fetched separately
                        score=neighbor.distance,
                        metadata={}  # Metadata needs to be fetched separately
                    ))
            
            self.logger.info(f"Similarity search returned {len(results)} results for org {self.org_id}")
            return results
            
        except Exception as e:
            self.logger.error(f"Similarity search failed for org {self.org_id}: {str(e)}", 
                             exc_info=True)
            raise VectorStoreException(f"Similarity search failed: {str(e)}")


class MultiTenantVectorStore:
    """
    Multi-tenant vector store manager with org-specific indexes.
    
    Features:
    - Separate Vertex AI Vector Search indexes per organization
    - Security metadata enforcement 
    - Automatic index creation and management
    - User/group level access filtering within organizations
    """
    
    def __init__(self):
        self.project_id = settings.gcp_project_id
        self.location = settings.gcp_location
        self.org_indexes: Dict[str, OrgVectorIndex] = {}
        self.logger = get_logger(__name__, "multi_tenant_vector_store")
    
    async def get_org_index(self, org_id: str) -> OrgVectorIndex:
        """Get or create an org-specific vector index."""
        if org_id not in self.org_indexes:
            self.org_indexes[org_id] = OrgVectorIndex(
                org_id=org_id,
                project_id=self.project_id,
                location=self.location
            )
            
            # Initialize the index
            success = await self.org_indexes[org_id].initialize()
            if not success:
                raise VectorStoreException(f"Failed to initialize index for org {org_id}")
        
        return self.org_indexes[org_id]
    
    async def upsert_chunks(self, chunks: List[Dict[str, Any]], org_id: str) -> Dict[str, Any]:
        """
        Upsert chunks to the appropriate org-specific index.
        
        Args:
            chunks: List of chunk dictionaries with content and metadata
            org_id: Organization identifier for routing to correct index
        """
        try:
            # Validate all chunks belong to the same org
            for chunk in chunks:
                chunk_org_id = chunk.get('metadata', {}).get('org_id')
                if chunk_org_id != org_id:
                    raise VectorStoreException(f"Chunk org_id mismatch: {chunk_org_id} != {org_id}")
            
            # Get org-specific index
            org_index = await self.get_org_index(org_id)
            
            # Upsert chunks
            result = await org_index.upsert_vectors(chunks)
            
            self.logger.info(f"Upserted chunks for org {org_id}: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to upsert chunks for org {org_id}: {str(e)}", exc_info=True)
            raise
    
    async def similarity_search(self,
                               query: str,
                               org_id: str,
                               user_id: str,
                               group_ids: List[str] = None,
                               top_k: int = 10,
                               filters: Dict[str, Any] = None) -> List[VectorSearchResult]:
        """
        Perform similarity search with multi-tenant security filtering.
        
        Args:
            query: Search query
            org_id: Organization ID for index routing
            user_id: User ID for access control
            group_ids: Optional group IDs for additional access
            top_k: Number of results to return
            filters: Additional metadata filters
        """
        try:
            # Get org-specific index
            org_index = await self.get_org_index(org_id)
            
            # Perform search with security filtering
            results = await org_index.similarity_search(
                query=query,
                top_k=top_k,
                user_id=user_id,
                group_ids=group_ids or [],
                filters=filters
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Similarity search failed for org {org_id}: {str(e)}", exc_info=True)
            raise
    
    def health_check(self) -> Dict[str, Any]:
        """Check health of all org indexes."""
        health_status = {
            "healthy": True,
            "total_orgs": len(self.org_indexes),
            "org_statuses": {}
        }
        
        for org_id, org_index in self.org_indexes.items():
            org_health = {
                "initialized": org_index._initialized,
                "has_index": org_index._index is not None,
                "has_endpoint": org_index._endpoint is not None
            }
            
            org_healthy = all(org_health.values())
            health_status["org_statuses"][org_id] = {
                **org_health,
                "healthy": org_healthy
            }
            
            if not org_healthy:
                health_status["healthy"] = False
        
        return health_status


# Global multi-tenant vector store instance
multi_tenant_vector_store = MultiTenantVectorStore()