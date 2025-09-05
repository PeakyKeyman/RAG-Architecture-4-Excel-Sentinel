"""
Security audit logging for the RAG Pipeline.

Provides comprehensive logging of security events, access attempts,
data operations, and system events for compliance and monitoring.
"""

import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List
from enum import Enum
from dataclasses import dataclass, asdict

from .safe_logging import get_safe_logger


class SecurityEventType(Enum):
    """Types of security events to log."""
    
    # Authentication events
    AUTH_SUCCESS = "auth_success"
    AUTH_FAILURE = "auth_failure"
    AUTH_INVALID_KEY = "auth_invalid_key"
    
    # Authorization events
    AUTHZ_GRANTED = "authz_granted"
    AUTHZ_DENIED = "authz_denied"
    AUTHZ_PRIVILEGE_ESCALATION = "authz_privilege_escalation"
    
    # Data access events
    DATA_ACCESS = "data_access"
    DATA_EXPORT = "data_export"
    DATA_DELETION = "data_deletion"
    DATA_MODIFICATION = "data_modification"
    
    # Multi-tenant security events
    TENANT_ISOLATION_VIOLATION = "tenant_isolation_violation"
    CROSS_ORG_ACCESS_ATTEMPT = "cross_org_access_attempt"
    
    # System security events
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    PII_DETECTED = "pii_detected"
    FILE_UPLOAD = "file_upload"
    
    # Configuration events
    CONFIG_CHANGE = "config_change"
    SECURITY_SETTING_CHANGE = "security_setting_change"


@dataclass
class SecurityEvent:
    """Structured security event for audit logging."""
    
    event_type: SecurityEventType
    timestamp: str
    event_id: str
    
    # Actor information
    user_id: Optional[str] = None
    org_id: Optional[str] = None
    group_id: Optional[str] = None
    client_ip: Optional[str] = None
    user_agent: Optional[str] = None
    
    # Resource information
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    resource_path: Optional[str] = None
    
    # Event details
    action: Optional[str] = None
    result: str = "unknown"  # success, failure, blocked, error
    details: Optional[Dict[str, Any]] = None
    
    # Security context
    access_level: Optional[str] = None
    permissions: Optional[List[str]] = None
    risk_score: int = 0  # 0-10, higher is more risky
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        data = asdict(self)
        # Convert enum to string
        data['event_type'] = self.event_type.value
        return data


class SecurityAuditor:
    """Central security audit logging system."""
    
    def __init__(self):
        self.logger = get_safe_logger(__name__)
        self._session_context = {}
    
    def set_session_context(self, 
                           user_id: str = None,
                           org_id: str = None,
                           group_id: str = None,
                           client_ip: str = None,
                           user_agent: str = None):
        """Set session context for subsequent audit logs."""
        self._session_context = {
            'user_id': user_id,
            'org_id': org_id,
            'group_id': group_id,
            'client_ip': client_ip,
            'user_agent': user_agent
        }
    
    def _create_event(self, 
                     event_type: SecurityEventType,
                     result: str = "unknown",
                     **kwargs) -> SecurityEvent:
        """Create a security event with session context."""
        event_id = f"audit_{int(time.time() * 1000)}_{hash(str(kwargs)) % 10000}"
        
        event = SecurityEvent(
            event_type=event_type,
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_id=event_id,
            result=result,
            **{**self._session_context, **kwargs}
        )
        
        return event
    
    def log_authentication_success(self, user_id: str, method: str = "api_key"):
        """Log successful authentication."""
        event = self._create_event(
            SecurityEventType.AUTH_SUCCESS,
            result="success",
            user_id=user_id,
            action=f"authenticate_{method}",
            details={"auth_method": method}
        )
        
        self.logger.info(
            f"Authentication successful for user {user_id}",
            extra={"security_event": event.to_dict()}
        )
    
    def log_authentication_failure(self, reason: str, client_ip: str = None):
        """Log failed authentication attempt."""
        event = self._create_event(
            SecurityEventType.AUTH_FAILURE,
            result="failure",
            client_ip=client_ip,
            action="authenticate_api_key",
            details={"failure_reason": reason},
            risk_score=7
        )
        
        self.logger.warning(
            f"Authentication failed: {reason}",
            extra={"security_event": event.to_dict()}
        )
    
    def log_authorization_denied(self, 
                                user_id: str,
                                resource_type: str,
                                resource_id: str,
                                required_permission: str):
        """Log authorization denial."""
        event = self._create_event(
            SecurityEventType.AUTHZ_DENIED,
            result="blocked",
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            action="access_resource",
            details={"required_permission": required_permission},
            risk_score=5
        )
        
        self.logger.warning(
            f"Access denied for user {user_id} to {resource_type}:{resource_id}",
            extra={"security_event": event.to_dict()}
        )
    
    def log_cross_tenant_access_attempt(self,
                                       user_id: str,
                                       user_org_id: str,
                                       target_org_id: str,
                                       resource_type: str):
        """Log attempt to access another organization's data."""
        event = self._create_event(
            SecurityEventType.CROSS_ORG_ACCESS_ATTEMPT,
            result="blocked",
            user_id=user_id,
            org_id=user_org_id,
            resource_type=resource_type,
            action="cross_tenant_access",
            details={
                "user_org": user_org_id,
                "target_org": target_org_id
            },
            risk_score=9  # Very high risk
        )
        
        self.logger.critical(
            f"Cross-tenant access attempt: user {user_id} from org {user_org_id} tried to access {target_org_id} data",
            extra={"security_event": event.to_dict()}
        )
    
    def log_data_access(self,
                       user_id: str,
                       resource_type: str,
                       resource_id: str,
                       access_level: str,
                       document_count: int = None):
        """Log data access events."""
        event = self._create_event(
            SecurityEventType.DATA_ACCESS,
            result="success",
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            access_level=access_level,
            action="query_data",
            details={
                "document_count": document_count,
                "access_level": access_level
            },
            risk_score=2 if access_level == "public" else 4
        )
        
        self.logger.info(
            f"Data access by user {user_id}: {resource_type}:{resource_id}",
            extra={"security_event": event.to_dict()}
        )
    
    def log_document_upload(self,
                           user_id: str,
                           org_id: str,
                           file_path: str,
                           file_size: int,
                           document_type: str,
                           has_pii: bool = False):
        """Log document upload events."""
        event = self._create_event(
            SecurityEventType.FILE_UPLOAD,
            result="success",
            user_id=user_id,
            org_id=org_id,
            resource_type="document",
            resource_path=file_path,
            action="upload_document",
            details={
                "file_size": file_size,
                "document_type": document_type,
                "has_pii": has_pii
            },
            risk_score=6 if has_pii else 3
        )
        
        self.logger.info(
            f"Document uploaded by user {user_id}: {file_path}",
            extra={"security_event": event.to_dict()}
        )
    
    def log_document_deletion(self,
                             user_id: str,
                             org_id: str,
                             document_id: str,
                             authorized: bool):
        """Log document deletion attempts."""
        event = self._create_event(
            SecurityEventType.DATA_DELETION,
            result="success" if authorized else "blocked",
            user_id=user_id,
            org_id=org_id,
            resource_type="document",
            resource_id=document_id,
            action="delete_document",
            details={"authorized": authorized},
            risk_score=8 if not authorized else 5
        )
        
        level = "info" if authorized else "warning"
        getattr(self.logger, level)(
            f"Document deletion {'authorized' if authorized else 'blocked'} for user {user_id}: {document_id}",
            extra={"security_event": event.to_dict()}
        )
    
    def log_rate_limit_exceeded(self,
                               user_id: str = None,
                               client_ip: str = None,
                               endpoint: str = None,
                               rate_limit: str = None):
        """Log rate limiting events."""
        event = self._create_event(
            SecurityEventType.RATE_LIMIT_EXCEEDED,
            result="blocked",
            user_id=user_id,
            client_ip=client_ip,
            resource_path=endpoint,
            action="api_request",
            details={"rate_limit": rate_limit},
            risk_score=6
        )
        
        self.logger.warning(
            f"Rate limit exceeded for {'user ' + user_id if user_id else 'IP ' + client_ip}",
            extra={"security_event": event.to_dict()}
        )
    
    def log_pii_detection(self,
                         user_id: str,
                         resource_type: str,
                         pii_types: List[str],
                         action_taken: str = "redacted"):
        """Log PII detection and handling."""
        event = self._create_event(
            SecurityEventType.PII_DETECTED,
            result="handled",
            user_id=user_id,
            resource_type=resource_type,
            action=action_taken,
            details={
                "pii_types": pii_types,
                "action_taken": action_taken
            },
            risk_score=7
        )
        
        self.logger.warning(
            f"PII detected in {resource_type} for user {user_id}: {pii_types}",
            extra={"security_event": event.to_dict()}
        )
    
    def log_suspicious_activity(self,
                               description: str,
                               user_id: str = None,
                               client_ip: str = None,
                               details: Dict[str, Any] = None):
        """Log suspicious activity."""
        event = self._create_event(
            SecurityEventType.SUSPICIOUS_ACTIVITY,
            result="detected",
            user_id=user_id,
            client_ip=client_ip,
            action="suspicious_behavior",
            details=details or {},
            risk_score=8
        )
        
        self.logger.critical(
            f"Suspicious activity detected: {description}",
            extra={"security_event": event.to_dict()}
        )
    
    def log_system_event(self, event_type: str, details: Dict[str, Any]):
        """Log general system security events."""
        event = self._create_event(
            SecurityEventType.CONFIG_CHANGE,
            result="success",
            action=event_type,
            details=details,
            risk_score=3
        )
        
        self.logger.info(
            f"System event: {event_type}",
            extra={"security_event": event.to_dict()}
        )


# Global security auditor instance
security_auditor = SecurityAuditor()