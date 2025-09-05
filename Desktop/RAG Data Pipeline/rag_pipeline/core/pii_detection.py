"""
PII Detection and Data Sanitization Utilities.

Provides functions to detect and redact personally identifiable information
and sensitive business data before logging or sending to external services.
"""

import re
from typing import Dict, List, Any, Optional, Union
import hashlib
from datetime import datetime


class PIIDetector:
    """Detects and redacts PII and sensitive business information."""
    
    # Common PII patterns
    EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    SSN_PATTERN = re.compile(r'\b\d{3}-?\d{2}-?\d{4}\b')
    PHONE_PATTERN = re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b')
    CREDIT_CARD_PATTERN = re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b')
    
    # Business sensitive patterns
    SALARY_PATTERN = re.compile(r'\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?')
    EMPLOYEE_ID_PATTERN = re.compile(r'\bEMP[-_]?\d{4,}\b', re.IGNORECASE)
    
    # Executive/Financial sensitive terms
    SENSITIVE_KEYWORDS = {
        'salary', 'compensation', 'bonus', 'stock_options', 'equity',
        'layoffs', 'termination', 'disciplinary', 'performance_review',
        'merger', 'acquisition', 'insider_trading', 'sec_filing',
        'earnings', 'revenue_projection', 'profit_margin', 'loss',
        'confidential', 'proprietary', 'trade_secret', 'patent_pending'
    }
    
    def __init__(self):
        """Initialize PII detector with configurable patterns."""
        self.redaction_map: Dict[str, str] = {}
    
    def _generate_consistent_hash(self, text: str, prefix: str = "REDACTED") -> str:
        """Generate consistent hash for the same PII value."""
        hash_obj = hashlib.md5(text.encode())
        short_hash = hash_obj.hexdigest()[:8]
        return f"[{prefix}_{short_hash}]"
    
    def detect_emails(self, text: str) -> List[str]:
        """Detect email addresses in text."""
        return self.EMAIL_PATTERN.findall(text)
    
    def detect_phone_numbers(self, text: str) -> List[str]:
        """Detect phone numbers in text."""
        return self.PHONE_PATTERN.findall(text)
    
    def detect_ssns(self, text: str) -> List[str]:
        """Detect Social Security Numbers in text."""
        return self.SSN_PATTERN.findall(text)
    
    def detect_sensitive_keywords(self, text: str) -> List[str]:
        """Detect sensitive business keywords."""
        text_lower = text.lower()
        found_keywords = []
        for keyword in self.SENSITIVE_KEYWORDS:
            if keyword in text_lower:
                found_keywords.append(keyword)
        return found_keywords
    
    def redact_pii(self, text: str, preserve_structure: bool = True) -> str:
        """
        Redact PII from text while optionally preserving structure.
        
        Args:
            text: Input text to redact
            preserve_structure: If True, maintains consistent redactions
        """
        if not text or not isinstance(text, str):
            return text
        
        redacted_text = text
        
        # Redact emails
        emails = self.detect_emails(redacted_text)
        for email in emails:
            if preserve_structure:
                replacement = self._generate_consistent_hash(email, "EMAIL")
            else:
                replacement = "[EMAIL_REDACTED]"
            redacted_text = redacted_text.replace(email, replacement)
        
        # Redact phone numbers
        phones = self.detect_phone_numbers(redacted_text)
        for phone in phones:
            if preserve_structure:
                replacement = self._generate_consistent_hash(phone, "PHONE")
            else:
                replacement = "[PHONE_REDACTED]"
            redacted_text = redacted_text.replace(phone, replacement)
        
        # Redact SSNs
        ssns = self.detect_ssns(redacted_text)
        for ssn in ssns:
            if preserve_structure:
                replacement = self._generate_consistent_hash(ssn, "SSN")
            else:
                replacement = "[SSN_REDACTED]"
            redacted_text = redacted_text.replace(ssn, replacement)
        
        # Redact credit cards
        cards = self.CREDIT_CARD_PATTERN.findall(redacted_text)
        for card in cards:
            if preserve_structure:
                replacement = self._generate_consistent_hash(card, "CARD")
            else:
                replacement = "[CARD_REDACTED]"
            redacted_text = redacted_text.replace(card, replacement)
        
        # Redact salary information
        salaries = self.SALARY_PATTERN.findall(redacted_text)
        for salary in salaries:
            if preserve_structure:
                replacement = self._generate_consistent_hash(salary, "AMOUNT")
            else:
                replacement = "[AMOUNT_REDACTED]"
            redacted_text = redacted_text.replace(salary, replacement)
        
        return redacted_text
    
    def sanitize_for_logging(self, data: Any) -> Any:
        """
        Sanitize data structure for safe logging.
        
        Recursively processes dictionaries, lists, and strings to remove PII.
        """
        if isinstance(data, str):
            return self.redact_pii(data)
        elif isinstance(data, dict):
            sanitized = {}
            for key, value in data.items():
                # Redact sensitive keys completely
                if key.lower() in {'password', 'api_key', 'secret', 'token', 'auth', 'credential'}:
                    sanitized[key] = "[REDACTED]"
                elif key.lower() in {'email', 'phone', 'ssn', 'social_security'}:
                    sanitized[key] = "[PII_REDACTED]" if value else value
                else:
                    sanitized[key] = self.sanitize_for_logging(value)
            return sanitized
        elif isinstance(data, list):
            return [self.sanitize_for_logging(item) for item in data]
        else:
            return data
    
    def sanitize_for_external_service(self, data: Any, service_name: str = "external") -> Any:
        """
        Sanitize data for external service transmission.
        
        More aggressive redaction for external services like Langsmith.
        """
        if isinstance(data, str):
            # For external services, be more aggressive
            redacted = self.redact_pii(data, preserve_structure=False)
            
            # Also redact document content beyond a certain length
            if len(redacted) > 500:  # Limit context size
                redacted = redacted[:500] + "[CONTENT_TRUNCATED]"
            
            return redacted
        elif isinstance(data, dict):
            sanitized = {}
            for key, value in data.items():
                # Skip certain keys entirely for external services
                if key.lower() in {'full_content', 'raw_content', 'document_text'}:
                    sanitized[key] = "[CONTENT_REDACTED_FOR_EXTERNAL_SERVICE]"
                elif key.lower() in {'metadata', 'user_id', 'org_id'}:
                    # Redact but keep structure
                    if isinstance(value, str):
                        sanitized[key] = self._generate_consistent_hash(value, "ID")
                    else:
                        sanitized[key] = self.sanitize_for_external_service(value, service_name)
                else:
                    sanitized[key] = self.sanitize_for_external_service(value, service_name)
            return sanitized
        elif isinstance(data, list):
            # Limit list size for external services
            limited_list = data[:3] if len(data) > 3 else data
            return [self.sanitize_for_external_service(item, service_name) for item in limited_list]
        else:
            return data
    
    def has_pii(self, text: str) -> bool:
        """Check if text contains PII without redacting."""
        if not text or not isinstance(text, str):
            return False
        
        return (
            bool(self.detect_emails(text)) or
            bool(self.detect_phone_numbers(text)) or
            bool(self.detect_ssns(text)) or
            bool(self.CREDIT_CARD_PATTERN.findall(text)) or
            bool(self.detect_sensitive_keywords(text))
        )
    
    def get_pii_summary(self, text: str) -> Dict[str, int]:
        """Get summary of PII types found in text."""
        if not text or not isinstance(text, str):
            return {}
        
        return {
            "emails": len(self.detect_emails(text)),
            "phones": len(self.detect_phone_numbers(text)),
            "ssns": len(self.detect_ssns(text)),
            "credit_cards": len(self.CREDIT_CARD_PATTERN.findall(text)),
            "sensitive_keywords": len(self.detect_sensitive_keywords(text))
        }


# Global PII detector instance
pii_detector = PIIDetector()