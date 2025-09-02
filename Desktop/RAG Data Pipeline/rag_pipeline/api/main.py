"""FastAPI main application for RAG Data Pipeline."""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
import asyncio

from ..core.config import settings
from ..core.logging_config import setup_logging, get_logger
from ..inference.pipeline import inference_pipeline
from .middleware import (
    AuthenticationMiddleware,
    ExceptionHandlingMiddleware,
    RequestLoggingMiddleware,
    RateLimitingMiddleware
)
from .routers import inference, vector_store, evaluation


# Setup logging
setup_logging()
logger = get_logger(__name__, "api_main")

# Create FastAPI app
app = FastAPI(
    title="RAG Data Pipeline API",
    description="High-performance RAG pipeline with hierarchical chunking, hybrid search, and knowledge base adjustment",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom middleware (order matters)
app.add_middleware(ExceptionHandlingMiddleware)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(RateLimitingMiddleware)
app.add_middleware(AuthenticationMiddleware)

# Include routers
app.include_router(inference.router)
app.include_router(vector_store.router)
app.include_router(evaluation.router)


@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    try:
        logger.info("Starting RAG Data Pipeline API...")
        
        # Initialize inference pipeline
        inference_pipeline.initialize()
        
        logger.info("RAG Data Pipeline API started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}", exc_info=True)
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    try:
        logger.info("Shutting down RAG Data Pipeline API...")
        
        # Perform any cleanup here
        
        logger.info("RAG Data Pipeline API shutdown complete")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {str(e)}", exc_info=True)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "RAG Data Pipeline API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": time.time(),
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
async def health_check():
    """Application health check endpoint."""
    try:
        # Check pipeline health
        pipeline_health = inference_pipeline.health_check()
        
        overall_health = {
            "status": "healthy" if pipeline_health["healthy"] else "unhealthy",
            "timestamp": time.time(),
            "version": "1.0.0",
            "pipeline": pipeline_health,
            "api": {
                "status": "healthy",
                "uptime_seconds": time.time() - startup_time if 'startup_time' in globals() else 0
            }
        }
        
        status_code = 200 if pipeline_health["healthy"] else 503
        
        return JSONResponse(
            status_code=status_code,
            content=overall_health
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }
        )


# Store startup time for uptime calculation
startup_time = time.time()


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "rag_pipeline.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        log_level=settings.log_level.lower(),
        reload=False  # Set to True for development
    )