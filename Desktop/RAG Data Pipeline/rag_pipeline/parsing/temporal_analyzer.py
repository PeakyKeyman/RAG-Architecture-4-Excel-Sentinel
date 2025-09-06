"""
Temporal analysis for document content.

Extracts temporal metadata from executive documents including:
- Document dates (creation, modification, referenced dates)
- Temporal relevance scoring
- Time-based content classification
- Temporal query understanding
"""

import re
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import calendar

from ..core.logging_config import get_logger


logger = get_logger(__name__, "temporal_analyzer")


class TemporalRelevance(Enum):
    """Temporal relevance categories for executive content."""
    CURRENT = "current"          # Within last 3 months
    RECENT = "recent"            # Within last year
    HISTORICAL = "historical"    # 1-3 years old
    ARCHIVED = "archived"        # Over 3 years old


@dataclass
class TemporalMetadata:
    """Extracted temporal information from document."""
    
    # Extracted dates
    document_date: Optional[datetime] = None
    creation_date: Optional[datetime] = None
    modification_date: Optional[datetime] = None
    referenced_dates: List[datetime] = None
    
    # Temporal classification
    relevance_category: TemporalRelevance = TemporalRelevance.CURRENT
    recency_score: float = 1.0  # 0-1, higher = more recent
    
    # Business temporal context
    fiscal_quarter: Optional[str] = None
    fiscal_year: Optional[int] = None
    reporting_period: Optional[str] = None
    
    # Temporal keywords found
    temporal_indicators: List[str] = None
    time_expressions: List[str] = None
    
    def __post_init__(self):
        if self.referenced_dates is None:
            self.referenced_dates = []
        if self.temporal_indicators is None:
            self.temporal_indicators = []
        if self.time_expressions is None:
            self.time_expressions = []


class TemporalAnalyzer:
    """
    Analyzes temporal aspects of executive documents.
    
    Optimized for business documents with focus on:
    - Financial reporting periods
    - Strategic planning timelines
    - Board meeting dates
    - Market event timestamps
    """
    
    # Date patterns for document parsing
    DATE_PATTERNS = [
        # Standard formats
        r'\b(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})\b',  # MM/DD/YYYY, MM-DD-YYYY
        r'\b(\d{4})[/\-.](\d{1,2})[/\-.](\d{1,2})\b',  # YYYY/MM/DD, YYYY-MM-DD
        r'\b(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2})\b',  # MM/DD/YY
        
        # Month names
        r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})\b',
        r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+(\d{1,2}),?\s+(\d{4})\b',
        r'\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})\b',
        
        # Quarter formats
        r'\bQ([1-4])\s+(\d{4})\b',
        r'\b([1-4])Q\s*(\d{4})\b',
        r'\b(First|Second|Third|Fourth)\s+Quarter\s+(\d{4})\b',
        
        # Fiscal year patterns
        r'\bFY\s*(\d{4})\b',
        r'\bFiscal\s+Year\s+(\d{4})\b',
        r'\bFiscal\s+(\d{4})\b',
    ]
    
    # Business temporal keywords
    TEMPORAL_KEYWORDS = {
        'current': ['current', 'present', 'now', 'today', 'this month', 'this quarter', 'this year'],
        'recent': ['recent', 'recently', 'latest', 'last month', 'last quarter', 'past year'],
        'future': ['future', 'upcoming', 'next month', 'next quarter', 'next year', 'projected', 'forecast'],
        'comparative': ['year-over-year', 'yoy', 'quarter-over-quarter', 'qoq', 'versus', 'compared to', 'vs'],
        'periodic': ['quarterly', 'annual', 'monthly', 'weekly', 'daily', 'periodic', 'regular']
    }
    
    # Executive-specific temporal expressions
    EXECUTIVE_TIME_EXPRESSIONS = [
        r'\b(year[- ]over[- ]year|y[o0]y)\b',
        r'\b(quarter[- ]over[- ]quarter|q[o0]q)\b',
        r'\b(same[- ]period[- ]last[- ]year|sply)\b',
        r'\b(trailing[- ]twelve[- ]months|ttm)\b',
        r'\b(run[- ]rate)\b',
        r'\b(as[- ]of|through)\s+\w+\s+\d{1,2},?\s+\d{4}\b'
    ]
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize temporal analyzer with configuration."""
        self.config = config or {}
        
        # Compile regex patterns for performance
        self.compiled_date_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.DATE_PATTERNS]
        self.compiled_time_expressions = [re.compile(pattern, re.IGNORECASE) for pattern in self.EXECUTIVE_TIME_EXPRESSIONS]
        
        # Configuration
        self.current_fiscal_year_start = self.config.get('fiscal_year_start', 'January')  # Can be customized
        self.relevance_decay_months = self.config.get('relevance_decay_months', 12)
        
    def analyze_document(self, content: str, file_path: str = None, 
                        creation_time: datetime = None) -> TemporalMetadata:
        """
        Extract temporal metadata from document content.
        
        Args:
            content: Document text content
            file_path: File path for additional context
            creation_time: File creation timestamp
            
        Returns:
            TemporalMetadata with extracted temporal information
        """
        metadata = TemporalMetadata()
        
        try:
            # Extract dates from content
            extracted_dates = self._extract_dates(content)
            
            if extracted_dates:
                # Most recent date is likely the document date
                metadata.referenced_dates = sorted(extracted_dates, reverse=True)
                metadata.document_date = metadata.referenced_dates[0]
            
            # Use creation time if provided
            if creation_time:
                metadata.creation_date = creation_time
                if not metadata.document_date:
                    metadata.document_date = creation_time
            
            # Extract temporal indicators
            metadata.temporal_indicators = self._extract_temporal_keywords(content)
            metadata.time_expressions = self._extract_time_expressions(content)
            
            # Determine fiscal context
            if metadata.document_date:
                metadata.fiscal_quarter = self._determine_fiscal_quarter(metadata.document_date)
                metadata.fiscal_year = self._determine_fiscal_year(metadata.document_date)
                metadata.reporting_period = self._determine_reporting_period(content, metadata.document_date)
            
            # Calculate relevance scores
            metadata.recency_score = self._calculate_recency_score(metadata.document_date)
            metadata.relevance_category = self._classify_temporal_relevance(metadata.document_date)
            
            logger.debug(
                f"Temporal analysis complete",
                extra={
                    "document_date": metadata.document_date.isoformat() if metadata.document_date else None,
                    "relevance_category": metadata.relevance_category.value,
                    "recency_score": metadata.recency_score,
                    "referenced_dates_count": len(metadata.referenced_dates),
                    "fiscal_quarter": metadata.fiscal_quarter
                }
            )
            
        except Exception as e:
            logger.warning(f"Temporal analysis failed: {str(e)}", exc_info=True)
        
        return metadata
    
    def _extract_dates(self, content: str) -> List[datetime]:
        """Extract all dates from content using multiple patterns."""
        dates = []
        
        for pattern in self.compiled_date_patterns:
            matches = pattern.finditer(content)
            
            for match in matches:
                try:
                    date_obj = self._parse_date_match(match)
                    if date_obj and self._is_reasonable_date(date_obj):
                        dates.append(date_obj)
                except (ValueError, TypeError):
                    continue
        
        # Remove duplicates and sort
        unique_dates = list(set(dates))
        return sorted(unique_dates, reverse=True)
    
    def _parse_date_match(self, match) -> Optional[datetime]:
        """Parse regex match into datetime object."""
        groups = match.groups()
        text = match.group(0).lower()
        
        try:
            # Handle quarter formats
            if 'q' in text or 'quarter' in text:
                if len(groups) >= 2:
                    quarter = int(groups[0]) if groups[0].isdigit() else self._quarter_name_to_number(groups[0])
                    year = int(groups[1])
                    if 1 <= quarter <= 4:
                        month = (quarter - 1) * 3 + 1  # Q1=Jan, Q2=Apr, Q3=Jul, Q4=Oct
                        return datetime(year, month, 1)
            
            # Handle fiscal year
            elif 'fy' in text or 'fiscal' in text:
                year = int(groups[0])
                return datetime(year, 1, 1)  # Approximate to start of year
            
            # Handle month names
            elif any(month in text for month in ['january', 'february', 'march', 'april', 'may', 'june',
                                                'july', 'august', 'september', 'october', 'november', 'december',
                                                'jan', 'feb', 'mar', 'apr', 'may', 'jun',
                                                'jul', 'aug', 'sep', 'oct', 'nov', 'dec']):
                if len(groups) >= 3:
                    month_str = groups[0] if not groups[0].isdigit() else groups[1]
                    day = int(groups[1]) if groups[0].isdigit() else int(groups[0])
                    year = int(groups[2])
                    
                    month = self._month_name_to_number(month_str)
                    if month and 1 <= day <= 31:
                        return datetime(year, month, day)
            
            # Handle numeric formats
            else:
                if len(groups) >= 3:
                    # Try different interpretations
                    nums = [int(g) for g in groups if g.isdigit()]
                    
                    if len(nums) >= 3:
                        # YYYY-MM-DD format
                        if nums[0] > 1900:
                            year, month, day = nums[0], nums[1], nums[2]
                        # MM/DD/YYYY format
                        elif nums[2] > 1900:
                            month, day, year = nums[0], nums[1], nums[2]
                        # MM/DD/YY format
                        else:
                            month, day, year = nums[0], nums[1], 2000 + nums[2] if nums[2] < 50 else 1900 + nums[2]
                        
                        if 1 <= month <= 12 and 1 <= day <= 31:
                            return datetime(year, month, day)
                            
        except (ValueError, TypeError, IndexError):
            pass
        
        return None
    
    def _quarter_name_to_number(self, quarter_name: str) -> Optional[int]:
        """Convert quarter name to number."""
        quarter_map = {
            'first': 1, 'second': 2, 'third': 3, 'fourth': 4,
            '1st': 1, '2nd': 2, '3rd': 3, '4th': 4
        }
        return quarter_map.get(quarter_name.lower())
    
    def _month_name_to_number(self, month_name: str) -> Optional[int]:
        """Convert month name to number."""
        month_name = month_name.lower()
        
        full_months = ['january', 'february', 'march', 'april', 'may', 'june',
                      'july', 'august', 'september', 'october', 'november', 'december']
        
        short_months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                       'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
        
        if month_name in full_months:
            return full_months.index(month_name) + 1
        elif month_name in short_months:
            return short_months.index(month_name) + 1
        
        return None
    
    def _is_reasonable_date(self, date_obj: datetime) -> bool:
        """Check if date is reasonable for business documents."""
        now = datetime.now()
        
        # Don't accept dates more than 50 years in the past or 5 years in the future
        min_date = now - timedelta(days=365 * 50)
        max_date = now + timedelta(days=365 * 5)
        
        return min_date <= date_obj <= max_date
    
    def _extract_temporal_keywords(self, content: str) -> List[str]:
        """Extract temporal keywords from content."""
        content_lower = content.lower()
        found_keywords = []
        
        for category, keywords in self.TEMPORAL_KEYWORDS.items():
            for keyword in keywords:
                if keyword in content_lower:
                    found_keywords.append(f"{category}:{keyword}")
        
        return found_keywords
    
    def _extract_time_expressions(self, content: str) -> List[str]:
        """Extract executive-specific time expressions."""
        expressions = []
        
        for pattern in self.compiled_time_expressions:
            matches = pattern.findall(content)
            expressions.extend(matches)
        
        return expressions
    
    def _determine_fiscal_quarter(self, date: datetime) -> str:
        """Determine fiscal quarter for given date."""
        if not date:
            return None
        
        # Assuming calendar year fiscal year (can be customized)
        quarter = (date.month - 1) // 3 + 1
        return f"Q{quarter}"
    
    def _determine_fiscal_year(self, date: datetime) -> int:
        """Determine fiscal year for given date."""
        if not date:
            return None
        
        # For calendar year fiscal year
        return date.year
    
    def _determine_reporting_period(self, content: str, date: datetime) -> Optional[str]:
        """Determine if document represents a specific reporting period."""
        content_lower = content.lower()
        
        # Check for common reporting periods
        if 'annual report' in content_lower or 'yearly' in content_lower:
            return f"Annual_{date.year}"
        elif any(q in content_lower for q in ['quarterly', 'q1', 'q2', 'q3', 'q4']):
            quarter = self._determine_fiscal_quarter(date)
            return f"{quarter}_{date.year}"
        elif 'monthly' in content_lower:
            return f"{date.strftime('%B_%Y')}"
        
        return None
    
    def _calculate_recency_score(self, document_date: datetime) -> float:
        """Calculate recency score (1.0 = most recent, 0.0 = oldest)."""
        if not document_date:
            return 0.5  # Neutral score for unknown dates
        
        now = datetime.now(timezone.utc)
        if document_date.tzinfo is None:
            document_date = document_date.replace(tzinfo=timezone.utc)
        
        # Calculate months difference
        months_diff = (now.year - document_date.year) * 12 + (now.month - document_date.month)
        
        # Apply exponential decay
        decay_rate = 1.0 / self.relevance_decay_months
        recency_score = max(0.0, min(1.0, 1.0 - (months_diff * decay_rate)))
        
        return recency_score
    
    def _classify_temporal_relevance(self, document_date: datetime) -> TemporalRelevance:
        """Classify document into temporal relevance category."""
        if not document_date:
            return TemporalRelevance.HISTORICAL
        
        now = datetime.now(timezone.utc)
        if document_date.tzinfo is None:
            document_date = document_date.replace(tzinfo=timezone.utc)
        
        months_diff = (now.year - document_date.year) * 12 + (now.month - document_date.month)
        
        if months_diff <= 3:
            return TemporalRelevance.CURRENT
        elif months_diff <= 12:
            return TemporalRelevance.RECENT
        elif months_diff <= 36:
            return TemporalRelevance.HISTORICAL
        else:
            return TemporalRelevance.ARCHIVED
    
    def analyze_query_temporality(self, query: str) -> Dict[str, Any]:
        """Analyze temporal aspects of user query."""
        query_lower = query.lower()
        temporal_info = {
            'has_temporal_intent': False,
            'temporal_keywords': [],
            'time_frame': None,
            'comparative': False,
            'recency_preference': 'current'  # Default preference
        }
        
        # Check for temporal keywords
        for category, keywords in self.TEMPORAL_KEYWORDS.items():
            found = [kw for kw in keywords if kw in query_lower]
            if found:
                temporal_info['temporal_keywords'].extend([(category, kw) for kw in found])
                temporal_info['has_temporal_intent'] = True
        
        # Check for comparative language
        if any(comp in query_lower for comp in ['versus', 'vs', 'compared to', 'difference', 'change']):
            temporal_info['comparative'] = True
        
        # Determine time frame preference
        if any(curr in query_lower for curr in ['current', 'now', 'today', 'this']):
            temporal_info['recency_preference'] = 'current'
        elif any(rec in query_lower for rec in ['recent', 'latest', 'last']):
            temporal_info['recency_preference'] = 'recent'
        elif any(hist in query_lower for hist in ['historical', 'history', 'past', 'over time']):
            temporal_info['recency_preference'] = 'historical'
        
        return temporal_info


# Global temporal analyzer instance
temporal_analyzer = TemporalAnalyzer()