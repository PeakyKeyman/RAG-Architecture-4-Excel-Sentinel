"""
Rule-based query complexity classifier for adaptive RAG routing.

Classifies executive queries into complexity levels to optimize retrieval strategy:
- Simple: Direct vector search (3x faster)
- Medium: Hybrid search with reranking (current default)  
- Complex: Full ensemble with multi-step reasoning
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from ..core.logging_config import get_logger


logger = get_logger(__name__, "query_classifier")


class QueryComplexity(Enum):
    """Query complexity levels for adaptive routing."""
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"


@dataclass
class QueryClassification:
    """Result of query complexity analysis."""
    complexity: QueryComplexity
    confidence: float
    reasoning: List[str]
    suggested_strategy: str
    estimated_tokens: int


class ExecutiveQueryClassifier:
    """
    Rule-based classifier optimized for executive and business queries.
    
    Designed for sub-millisecond classification with high accuracy on
    business document queries typical in executive environments.
    """
    
    # Simple query patterns (direct vector search sufficient)
    SIMPLE_PATTERNS = {
        'factual_lookup': [
            r'^(what|who|when|where|which)\s+is\s+',
            r'^define\s+',
            r'^(list|show)\s+(me\s+)?the\s+',
            r'^\w+(\s+\w+){0,2}\?\s*$',  # Very short questions
        ],
        'status_check': [
            r'status\s+of\s+',
            r'current\s+\w+',
            r'latest\s+\w+',
        ]
    }
    
    # Complex query patterns (full ensemble needed)
    COMPLEX_PATTERNS = {
        'multi_part': [
            r'\band\b.*\b(also|additionally|furthermore)\b',
            r'\b(compare|contrast|versus|vs\.?|difference)\b',
            r'\b(analyze|analysis|evaluate|assessment)\b',
            r'\b(strategy|strategic|recommendation)\b.*\b(options|alternatives)\b',
        ],
        'analytical': [
            r'\b(why|how)\s+(did|does|do|can|could|should|would)\b',
            r'\b(impact|effect|consequence|result)\b.*\b(of|from|due to)\b',
            r'\b(trend|trending|pattern|correlation)\b',
            r'\b(forecast|predict|projection|outlook)\b',
        ],
        'temporal_complex': [
            r'\b(since|from|between)\b.*\b(and|to|until)\b.*\b(now|today|current)\b',
            r'\b(year over year|quarterly|q1|q2|q3|q4)\b.*\b(comparison|change|growth)\b',
            r'\b(historical|timeline|evolution|progression)\b',
        ],
        'executive_reasoning': [
            r'\b(should we|what if|suppose|assuming)\b',
            r'\b(pros and cons|advantages.*disadvantages|benefits.*risks)\b',
            r'\b(roi|return on investment|cost.*benefit)\b',
            r'\b(market.*position|competitive.*advantage)\b',
        ]
    }
    
    # Keywords that increase complexity
    COMPLEXITY_KEYWORDS = {
        'high': [
            'strategy', 'strategic', 'analyze', 'analysis', 'compare', 'contrast',
            'evaluate', 'assessment', 'recommendation', 'optimize', 'maximize',
            'competitive', 'market', 'roi', 'investment', 'risk', 'opportunity'
        ],
        'medium': [
            'summary', 'overview', 'explain', 'describe', 'details', 'information',
            'report', 'data', 'metrics', 'performance', 'results', 'update'
        ]
    }
    
    # Executive domain indicators
    EXECUTIVE_DOMAINS = {
        'financial': ['revenue', 'profit', 'ebitda', 'cash flow', 'budget', 'cost', 'expense'],
        'strategic': ['vision', 'mission', 'goals', 'objectives', 'okr', 'kpi', 'initiative'],
        'operational': ['process', 'efficiency', 'productivity', 'workflow', 'operations'],
        'governance': ['compliance', 'policy', 'regulation', 'audit', 'risk', 'governance'],
        'hr': ['talent', 'hiring', 'retention', 'culture', 'leadership', 'development'],
        'market': ['customer', 'market', 'competitor', 'industry', 'segment', 'positioning']
    }
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize classifier with optional configuration."""
        self.config = config or {}
        self.enable_logging = self.config.get('enable_classifier_logging', False)
        
        # Compile regex patterns for performance
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for better performance."""
        self.compiled_simple = {}
        self.compiled_complex = {}
        
        for category, patterns in self.SIMPLE_PATTERNS.items():
            self.compiled_simple[category] = [re.compile(p, re.IGNORECASE) for p in patterns]
        
        for category, patterns in self.COMPLEX_PATTERNS.items():
            self.compiled_complex[category] = [re.compile(p, re.IGNORECASE) for p in patterns]
    
    def classify(self, query: str) -> QueryClassification:
        """
        Classify query complexity using rule-based analysis.
        
        Args:
            query: User query string
            
        Returns:
            QueryClassification with complexity level and reasoning
        """
        if not query or not query.strip():
            return QueryClassification(
                complexity=QueryComplexity.SIMPLE,
                confidence=1.0,
                reasoning=["Empty query"],
                suggested_strategy="direct_vector",
                estimated_tokens=0
            )
        
        query = query.strip()
        reasoning = []
        complexity_score = 0
        
        # Basic metrics
        word_count = len(query.split())
        sentence_count = len([s for s in re.split(r'[.!?]+', query) if s.strip()])
        question_marks = query.count('?')
        
        reasoning.append(f"Word count: {word_count}, Sentences: {sentence_count}")
        
        # Check for simple patterns first
        simple_matches = self._match_patterns(query, self.compiled_simple)
        if simple_matches:
            reasoning.extend([f"Simple pattern: {match}" for match in simple_matches])
            complexity_score -= 2
        
        # Check for complex patterns
        complex_matches = self._match_patterns(query, self.compiled_complex)
        if complex_matches:
            reasoning.extend([f"Complex pattern: {match}" for match in complex_matches])
            complexity_score += len(complex_matches) * 2
        
        # Word count scoring
        if word_count <= 5:
            complexity_score -= 1
            reasoning.append("Very short query")
        elif word_count <= 10:
            complexity_score -= 0.5
            reasoning.append("Short query")
        elif word_count >= 20:
            complexity_score += 1
            reasoning.append("Long query")
        
        # Multiple sentences increase complexity
        if sentence_count > 1:
            complexity_score += sentence_count * 0.5
            reasoning.append(f"Multi-sentence query ({sentence_count})")
        
        # Multiple questions increase complexity
        if question_marks > 1:
            complexity_score += question_marks * 0.5
            reasoning.append(f"Multiple questions ({question_marks})")
        
        # Keyword analysis
        high_keywords = sum(1 for kw in self.COMPLEXITY_KEYWORDS['high'] if kw in query.lower())
        medium_keywords = sum(1 for kw in self.COMPLEXITY_KEYWORDS['medium'] if kw in query.lower())
        
        if high_keywords > 0:
            complexity_score += high_keywords * 1.5
            reasoning.append(f"High complexity keywords: {high_keywords}")
        
        if medium_keywords > 0:
            complexity_score += medium_keywords * 0.5
            reasoning.append(f"Medium complexity keywords: {medium_keywords}")
        
        # Executive domain detection
        domain_matches = self._detect_executive_domains(query)
        if len(domain_matches) > 1:
            complexity_score += 1
            reasoning.append(f"Multiple domains: {domain_matches}")
        
        # Determine final complexity
        if complexity_score <= -1:
            complexity = QueryComplexity.SIMPLE
            strategy = "direct_vector"
            confidence = 0.9
        elif complexity_score >= 2:
            complexity = QueryComplexity.COMPLEX
            strategy = "full_ensemble"
            confidence = 0.8
        else:
            complexity = QueryComplexity.MEDIUM
            strategy = "hybrid_rerank"
            confidence = 0.7
        
        # Adjust confidence based on pattern matches
        if simple_matches and complexity == QueryComplexity.SIMPLE:
            confidence = 0.95
        if complex_matches and complexity == QueryComplexity.COMPLEX:
            confidence = 0.9
        
        classification = QueryClassification(
            complexity=complexity,
            confidence=confidence,
            reasoning=reasoning,
            suggested_strategy=strategy,
            estimated_tokens=word_count * 1.3  # Rough token estimation
        )
        
        if self.enable_logging:
            logger.debug(
                f"Query classified as {complexity.value}",
                extra={
                    "query_length": len(query),
                    "word_count": word_count,
                    "complexity_score": complexity_score,
                    "confidence": confidence,
                    "strategy": strategy
                }
            )
        
        return classification
    
    def _match_patterns(self, query: str, pattern_dict: Dict) -> List[str]:
        """Match query against compiled regex patterns."""
        matches = []
        for category, patterns in pattern_dict.items():
            for pattern in patterns:
                if pattern.search(query):
                    matches.append(category)
                    break  # Only count each category once
        return matches
    
    def _detect_executive_domains(self, query: str) -> List[str]:
        """Detect executive business domains in query."""
        query_lower = query.lower()
        domains = []
        
        for domain, keywords in self.EXECUTIVE_DOMAINS.items():
            if any(kw in query_lower for kw in keywords):
                domains.append(domain)
        
        return domains
    
    def get_classification_stats(self, queries: List[str]) -> Dict[str, int]:
        """Get classification statistics for a list of queries."""
        stats = {complexity.value: 0 for complexity in QueryComplexity}
        
        for query in queries:
            classification = self.classify(query)
            stats[classification.complexity.value] += 1
        
        return stats


# Global classifier instance
query_classifier = ExecutiveQueryClassifier()