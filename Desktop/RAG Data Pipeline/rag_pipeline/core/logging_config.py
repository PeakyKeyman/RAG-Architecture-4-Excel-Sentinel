"""JSON logging configuration for RAG pipeline."""

import json
import logging
import sys
from datetime import datetime
from typing import Any, Dict, Optional
from pythonjsonlogger import jsonlogger

from .config import settings


class RAGJSONFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter for RAG pipeline logs."""
    
    def add_fields(self, log_record: Dict[str, Any], record: logging.LogRecord, message_dict: Dict[str, Any]) -> None:
        super().add_fields(log_record, record, message_dict)
        
        # Add standard fields
        log_record['timestamp'] = datetime.utcnow().isoformat()
        log_record['level'] = record.levelname
        log_record['logger'] = record.name
        log_record['component'] = getattr(record, 'component', 'unknown')
        
        # Add request context if available
        if hasattr(record, 'request_id'):
            log_record['request_id'] = record.request_id
        
        if hasattr(record, 'user_id'):
            log_record['user_id'] = record.user_id


def setup_logging() -> None:
    """Setup JSON logging configuration."""
    
    # Create formatter
    formatter = RAGJSONFormatter(
        fmt='%(timestamp)s %(level)s %(logger)s %(component)s %(message)s'
    )
    
    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    handlers = [console_handler]
    
    # Setup file handler if specified
    if settings.log_file:
        file_handler = logging.FileHandler(settings.log_file)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        handlers=handlers,
        force=True
    )
    
    # Suppress verbose third-party logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("google").setLevel(logging.WARNING)


def get_logger(name: str, component: str = "unknown") -> logging.LoggerAdapter:
    """Get a logger with component context."""
    logger = logging.getLogger(name)
    return logging.LoggerAdapter(logger, {'component': component})


def log_exception(
    logger: logging.LoggerAdapter, 
    exception: Exception, 
    context: Optional[Dict[str, Any]] = None
) -> None:
    """Log an exception with full context."""
    from .exceptions import RAGPipelineException
    
    if isinstance(exception, RAGPipelineException):
        log_data = exception.to_dict()
        if context:
            log_data['context'] = context
        logger.error("Pipeline exception occurred", extra=log_data)
    else:
        logger.error(
            f"Unexpected exception: {str(exception)}",
            extra={
                'exception_type': type(exception).__name__,
                'exception_message': str(exception),
                'context': context or {}
            },
            exc_info=True
        )


def log_performance(
    logger: logging.LoggerAdapter,
    operation: str,
    duration_ms: float,
    success: bool = True,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Log performance metrics."""
    logger.info(
        f"Performance metric: {operation}",
        extra={
            'operation': operation,
            'duration_ms': duration_ms,
            'success': success,
            'metadata': metadata or {}
        }
    )