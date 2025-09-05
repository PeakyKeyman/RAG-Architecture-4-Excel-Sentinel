"""
Safe logging utilities with PII detection.

This module provides logging functions that automatically sanitize
data before writing to logs to prevent PII exposure.
"""

import logging
from typing import Any, Dict, Optional, Union


class SafeLogger:
    """Logger wrapper that sanitizes data before logging."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self._pii_detector = None
    
    def _get_pii_detector(self):
        """Lazy load PII detector to avoid circular imports."""
        if self._pii_detector is None:
            try:
                from .pii_detection import pii_detector
                self._pii_detector = pii_detector
            except ImportError:
                # Fallback if PII detection not available
                self._pii_detector = None
        return self._pii_detector
    
    def _sanitize_extra(self, extra: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Sanitize extra logging data."""
        if not extra:
            return extra
        
        detector = self._get_pii_detector()
        if detector:
            return detector.sanitize_for_logging(extra)
        
        # Basic sanitization fallback
        sanitized = {}
        for key, value in extra.items():
            if key.lower() in {'password', 'api_key', 'secret', 'token', 'auth'}:
                sanitized[key] = "[REDACTED]"
            elif isinstance(value, str) and len(value) > 1000:
                sanitized[key] = value[:1000] + "[TRUNCATED]"
            else:
                sanitized[key] = value
        return sanitized
    
    def _sanitize_message(self, msg: str) -> str:
        """Sanitize log message."""
        detector = self._get_pii_detector()
        if detector and isinstance(msg, str):
            return detector.redact_pii(msg)
        return msg
    
    def debug(self, msg: Any, extra: Optional[Dict[str, Any]] = None, **kwargs):
        """Debug level logging with PII sanitization."""
        self.logger.debug(
            self._sanitize_message(str(msg)), 
            extra=self._sanitize_extra(extra),
            **kwargs
        )
    
    def info(self, msg: Any, extra: Optional[Dict[str, Any]] = None, **kwargs):
        """Info level logging with PII sanitization."""
        self.logger.info(
            self._sanitize_message(str(msg)),
            extra=self._sanitize_extra(extra),
            **kwargs
        )
    
    def warning(self, msg: Any, extra: Optional[Dict[str, Any]] = None, **kwargs):
        """Warning level logging with PII sanitization.""" 
        self.logger.warning(
            self._sanitize_message(str(msg)),
            extra=self._sanitize_extra(extra),
            **kwargs
        )
    
    def error(self, msg: Any, extra: Optional[Dict[str, Any]] = None, exc_info: bool = False, **kwargs):
        """Error level logging with PII sanitization."""
        self.logger.error(
            self._sanitize_message(str(msg)),
            extra=self._sanitize_extra(extra),
            exc_info=exc_info,
            **kwargs
        )
    
    def critical(self, msg: Any, extra: Optional[Dict[str, Any]] = None, **kwargs):
        """Critical level logging with PII sanitization."""
        self.logger.critical(
            self._sanitize_message(str(msg)),
            extra=self._sanitize_extra(extra),
            **kwargs
        )
    
    def bind(self, **kwargs) -> 'SafeLogger':
        """Create a new logger with bound context (for structured logging)."""
        # For now, just return self - in a full implementation,
        # this would create a new logger with the bound context
        return self


def get_safe_logger(name: str, component: str = None) -> SafeLogger:
    """
    Get a safe logger instance that automatically sanitizes PII.
    
    Args:
        name: Logger name (usually __name__)
        component: Optional component name for structured logging
    
    Returns:
        SafeLogger instance that sanitizes PII before logging
    """
    logger = logging.getLogger(name)
    return SafeLogger(logger)