"""Security middleware for the RAG API."""

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add security headers to all responses."""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        
        # Content Security Policy for API
        response.headers["Content-Security-Policy"] = (
            "default-src 'none'; "
            "script-src 'self'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; "
            "connect-src 'self'; "
            "font-src 'self'; "
            "base-uri 'self'; "
            "form-action 'self'"
        )
        
        # Remove server identification
        if "Server" in response.headers:
            del response.headers["Server"]
        
        return response


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Middleware to limit request body size."""
    
    def __init__(self, app: ASGIApp, max_size: int = 50 * 1024 * 1024):  # 50MB default
        super().__init__(app)
        self.max_size = max_size
    
    async def dispatch(self, request: Request, call_next):
        # Check content length header
        content_length = request.headers.get("content-length")
        if content_length:
            if int(content_length) > self.max_size:
                return Response(
                    content=f"Request too large. Maximum size: {self.max_size} bytes",
                    status_code=413,
                    headers={"Content-Type": "text/plain"}
                )
        
        # For streaming requests, we'd need to implement body size checking
        # during the request processing - this is a basic implementation
        response = await call_next(request)
        return response