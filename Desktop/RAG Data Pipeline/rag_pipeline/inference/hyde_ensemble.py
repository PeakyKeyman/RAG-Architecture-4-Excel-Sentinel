"""Ensemble HyDE (Hypothetical Document Embeddings) implementation."""

import asyncio
import time
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import google.generativeai as genai
from google.generativeai.types import GenerateContentResponse

from ..core.config import settings
from ..core.exceptions import HyDEException
from ..core.logging_config import get_logger, log_performance


class HyDEGenerator:
    """Single HyDE generator using Gemini."""
    
    def __init__(self, model_name: str, generation_config: Dict[str, Any], prompt_template: str):
        self.model_name = model_name
        self.generation_config = generation_config
        self.prompt_template = prompt_template
        self.model = None
        
    def _initialize(self) -> None:
        """Initialize the Gemini model."""
        if self.model is None:
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=self.generation_config
            )
    
    def generate(self, query: str) -> str:
        """Generate a hypothetical document for the query."""
        self._initialize()
        
        prompt = self.prompt_template.format(query=query)
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip() if response.text else ""
            
        except Exception as e:
            raise HyDEException(
                f"Failed to generate hypothetical document: {str(e)}",
                component="hyde",
                error_code="GENERATION_FAILED",
                details={
                    "model": self.model_name,
                    "query": query[:100]
                }
            )


class EnsembleHyDE:
    """Ensemble HyDE implementation with multiple Gemini model variants."""
    
    def __init__(self, ensemble_size: int = None):
        self.logger = get_logger(__name__, "hyde_ensemble")
        self.ensemble_size = ensemble_size or settings.hyde_ensemble_size
        
        # Configure Gemini API
        if not hasattr(genai, '_client'):
            genai.configure(api_key=settings.gemini_api_key)
        
        # Define different generation configurations for diversity
        self.generation_configs = [
            {
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 40,
                "max_output_tokens": 512
            },
            {
                "temperature": 0.9,
                "top_p": 0.9,
                "top_k": 40,
                "max_output_tokens": 512
            },
            {
                "temperature": 0.5,
                "top_p": 0.7,
                "top_k": 20,
                "max_output_tokens": 512
            },
            {
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 50,
                "max_output_tokens": 512
            },
            {
                "temperature": 0.3,
                "top_p": 0.6,
                "top_k": 10,
                "max_output_tokens": 512
            }
        ]
        
        # Define different prompt templates for diversity
        self.prompt_templates = [
            """Write a comprehensive answer to this question: {query}
            
            Provide detailed information that would be found in a high-quality document addressing this topic.""",
            
            """Given this query: {query}
            
            Generate a detailed passage that contains relevant information, facts, and context that would help answer this question thoroughly.""",
            
            """Create an informative document excerpt that addresses: {query}
            
            Include specific details, examples, and explanations that would be valuable for understanding this topic.""",
            
            """Write a detailed response to: {query}
            
            Focus on providing comprehensive information with supporting details and context.""",
            
            """Generate relevant content for the query: {query}
            
            Provide an in-depth explanation with facts, examples, and comprehensive coverage of the topic."""
        ]
        
        # Create generators
        self.generators = self._create_generators()
    
    def _create_generators(self) -> List[HyDEGenerator]:
        """Create HyDE generators with different configurations."""
        generators = []
        
        for i in range(self.ensemble_size):
            config_idx = i % len(self.generation_configs)
            template_idx = i % len(self.prompt_templates)
            
            generator = HyDEGenerator(
                model_name=settings.gemini_model,
                generation_config=self.generation_configs[config_idx],
                prompt_template=self.prompt_templates[template_idx]
            )
            generators.append(generator)
        
        return generators
    
    def generate_hypothetical_documents(self, query: str) -> List[str]:
        """Generate multiple hypothetical documents using ensemble approach."""
        if not query.strip():
            raise HyDEException(
                "Empty query provided for HyDE generation",
                component="hyde_ensemble",
                error_code="EMPTY_QUERY"
            )
        
        try:
            start_time = time.time()
            
            # Generate documents in parallel
            hypothetical_docs = []
            failed_generations = 0
            failure_details = []
            
            with ThreadPoolExecutor(max_workers=self.ensemble_size) as executor:
                # Submit all generation tasks
                future_to_generator = {
                    executor.submit(generator.generate, query): i 
                    for i, generator in enumerate(self.generators)
                }
                
                # Collect results
                for future in as_completed(future_to_generator):
                    generator_idx = future_to_generator[future]
                    
                    try:
                        doc = future.result(timeout=30)
                        if doc and doc.strip():
                            hypothetical_docs.append(doc)
                        else:
                            failed_generations += 1
                            failure_details.append(f"Generator {generator_idx}: empty document")
                            self.logger.warning(f"Generator {generator_idx} returned empty document")
                            
                    except Exception as e:
                        failed_generations += 1
                        error_msg = f"Generator {generator_idx}: {str(e)}"
                        failure_details.append(error_msg)
                        self.logger.error(
                            f"Generator {generator_idx} failed: {str(e)}",
                            extra={"generator_idx": generator_idx, "query": query[:50]}
                        )
            
            # Ensure we have at least one document
            if not hypothetical_docs:
                raise HyDEException(
                    "All HyDE generators failed to produce documents",
                    component="hyde_ensemble",
                    error_code="ALL_GENERATORS_FAILED",
                    details={
                        "failed_count": failed_generations,
                        "failure_details": failure_details
                    }
                )
            
            duration = (time.time() - start_time) * 1000
            
            log_performance(
                self.logger,
                "generate_hypothetical_documents",
                duration,
                success=failed_generations == 0,
                metadata={
                    "query_length": len(query),
                    "ensemble_size": self.ensemble_size,
                    "successful_docs": len(hypothetical_docs),
                    "failed_generations": failed_generations,
                    "avg_doc_length": sum(len(doc) for doc in hypothetical_docs) / len(hypothetical_docs)
                }
            )
            
            self.logger.info(
                f"Generated {len(hypothetical_docs)} hypothetical documents",
                extra={
                    "successful": len(hypothetical_docs),
                    "failed": failed_generations,
                    "query": query[:100]
                }
            )
            
            return hypothetical_docs
            
        except HyDEException:
            raise
        except Exception as e:
            raise HyDEException(
                f"Unexpected error in HyDE generation: {str(e)}",
                component="hyde_ensemble",
                error_code="HYDE_UNEXPECTED_ERROR",
                details={"query": query[:100]}
            )
    
    async def generate_hypothetical_documents_async(self, query: str) -> List[str]:
        """Asynchronously generate hypothetical documents."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate_hypothetical_documents, query)
    
    def get_ensemble_info(self) -> Dict[str, Any]:
        """Get information about the HyDE ensemble."""
        return {
            "ensemble_size": self.ensemble_size,
            "model": settings.gemini_model,
            "generators_count": len(self.generators),
            "config_variants": len(self.generation_configs),
            "prompt_variants": len(self.prompt_templates)
        }


# Global HyDE ensemble instance
hyde_ensemble = EnsembleHyDE()