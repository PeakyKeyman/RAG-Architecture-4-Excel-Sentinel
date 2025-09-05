"""Authentication and error handling middleware for RAG API."""

import time
import uuid
import hmac
from typing import Callable, Dict, Any
from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from ..core.config import settings
from ..core.exceptions import RAGPipelineException
from ..core.logging_config import get_logger, log_exception


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """Simple API key authentication middleware."""
    
    def __init__(self, app):
        super().__init__(app)
        self.logger = get_logger(__name__, "auth_middleware")
    
    def _secure_compare(self, a: str, b: str) -> bool:
        """Constant-time string comparison to prevent timing attacks."""
        return hmac.compare_digest(a.encode(), b.encode())
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip auth for health and docs endpoints
        if request.url.path in ["/health", "/docs", "/redoc", "/openapi.json"]:
            return await call_next(request)
        
        # Check API key using constant-time comparison
        api_key = request.headers.get("X-API-Key")
        if not api_key or not self._secure_compare(api_key, settings.api_key):
            self.logger.warning(
                f"Unauthorized access attempt from {request.client.host}",
                extra={"path": request.url.path, "client_host": request.client.host}
            )
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"error": "Invalid API key", "code": "UNAUTHORIZED"}
            )
        
        return await call_next(request)


class ExceptionHandlingMiddleware(BaseHTTPMiddleware):
    """Global exception handling middleware."""
    
    def __init__(self, app):
        super().__init__(app)
        self.logger = get_logger(__name__, "exception_middleware")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            return await call_next(request)
        except RAGPipelineException as e:
            log_exception(self.logger, e, {"path": request.url.path})
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=e.to_dict()
            )
        except HTTPException:
            raise
        except Exception as e:
            log_exception(self.logger, e, {"path": request.url.path})
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": "Internal server error",
                    "code": "INTERNAL_ERROR",
                    "message": str(e)
                }
            )


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Request/response logging middleware."""
    
    def __init__(self, app):
        super().__init__(app)
        self.logger = get_logger(__name__, "request_middleware")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Add request ID to request state
        request.state.request_id = request_id
        
        self.logger.info(
            f"Request started: {request.method} {request.url.path}",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "client_host": request.client.host if request.client else None
            }
        )
        
        response = await call_next(request)
        
        duration = (time.time() - start_time) * 1000
        
        self.logger.info(
            f"Request completed: {response.status_code}",
            extra={
                "request_id": request_id,
                "status_code": response.status_code,
                "duration_ms": duration
            }
        )
        
        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id
        
        return response


class RateLimitingMiddleware(BaseHTTPMiddleware):
    """Simple rate limiting middleware."""
    
    def __init__(self, app):
        super().__init__(app)
        self.logger = get_logger(__name__, "rate_limit_middleware")
        self.client_requests: Dict[str, list] = {}
        self.max_requests = settings.max_concurrent_requests
        self.window_seconds = 60
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        client_ip = request.client.host if request.client else "unknown"
        current_time = time.time()
        
        # Clean old requests
        if client_ip in self.client_requests:
            self.client_requests[client_ip] = [
                req_time for req_time in self.client_requests[client_ip]
                if current_time - req_time < self.window_seconds
            ]
        else:
            self.client_requests[client_ip] = []
        
        # Check rate limit
        if len(self.client_requests[client_ip]) >= self.max_requests:
            self.logger.warning(
                f"Rate limit exceeded for {client_ip}",
                extra={"client_ip": client_ip, "request_count": len(self.client_requests[client_ip])}
            )
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "code": "RATE_LIMIT_EXCEEDED",
                    "retry_after": self.window_seconds
                }
            )
        
        # Add current request
        self.client_requests[client_ip].append(current_time)
        
        return await call_next(request)