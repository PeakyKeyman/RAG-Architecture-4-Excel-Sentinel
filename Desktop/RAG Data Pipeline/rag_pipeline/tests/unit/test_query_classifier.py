"""Unit tests for rule-based query complexity classifier."""

import pytest
from datetime import datetime

from ...inference.query_classifier import (
    ExecutiveQueryClassifier,
    QueryComplexity,
    QueryClassification
)


class TestExecutiveQueryClassifier:
    """Test cases for ExecutiveQueryClassifier."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.classifier = ExecutiveQueryClassifier()
    
    def test_simple_queries(self):
        """Test classification of simple queries."""
        simple_queries = [
            "What is EBITDA?",
            "Who is the CEO?",
            "Define revenue",
            "Show me the KPIs",
            "Current status"
        ]
        
        for query in simple_queries:
            classification = self.classifier.classify(query)
            assert classification.complexity == QueryComplexity.SIMPLE
            assert classification.confidence >= 0.8
            assert classification.suggested_strategy == "direct_vector"
    
    def test_complex_queries(self):
        """Test classification of complex queries."""
        complex_queries = [
            "Compare our Q3 performance to Q2 and analyze the key drivers of revenue growth",
            "What strategic initiatives should we prioritize for next year and what are the risks?",
            "How has our market position changed since the merger and what competitive advantages do we have?",
            "Analyze the impact of our pricing strategy on customer retention and recommend alternatives",
            "Should we invest in AI technology and what would be the ROI and implementation timeline?"
        ]
        
        for query in complex_queries:
            classification = self.classifier.classify(query)
            assert classification.complexity == QueryComplexity.COMPLEX
            assert classification.suggested_strategy == "full_ensemble"
    
    def test_medium_queries(self):
        """Test classification of medium complexity queries."""
        medium_queries = [
            "Summarize our quarterly financial results",
            "What are the main risks in our business?",
            "Explain our customer acquisition strategy",
            "Describe our competitive landscape",
            "Update on digital transformation progress"
        ]
        
        for query in medium_queries:
            classification = self.classifier.classify(query)
            assert classification.complexity == QueryComplexity.MEDIUM
            assert classification.suggested_strategy == "hybrid_rerank"
    
    def test_executive_domain_detection(self):
        """Test detection of executive business domains."""
        test_cases = [
            ("What's our revenue growth?", ["financial"]),
            ("Strategic vision for 2024", ["strategic"]),
            ("Board meeting agenda", ["governance"]),
            ("Customer retention metrics", ["market"]),
            ("Talent acquisition plan", ["hr"]),
            ("Process optimization results", ["operational"])
        ]
        
        for query, expected_domains in test_cases:
            domains = self.classifier._detect_executive_domains(query)
            assert any(domain in domains for domain in expected_domains)
    
    def test_temporal_patterns(self):
        """Test detection of temporal complexity patterns."""
        temporal_queries = [
            "Compare Q1 2023 to Q1 2024 performance",
            "Year-over-year revenue growth analysis",
            "Historical trends in customer satisfaction",
            "Quarterly results since the acquisition"
        ]
        
        for query in temporal_queries:
            classification = self.classifier.classify(query)
            # Temporal queries should typically be complex
            assert classification.complexity in [QueryComplexity.MEDIUM, QueryComplexity.COMPLEX]
    
    def test_empty_and_edge_cases(self):
        """Test edge cases and error handling."""
        edge_cases = [
            "",
            "   ",
            "?",
            "a",
            "This is a very long query that goes on and on and includes many words but doesn't really contain complex business logic or multiple questions or analytical requirements"
        ]
        
        for query in edge_cases:
            classification = self.classifier.classify(query)
            assert isinstance(classification, QueryClassification)
            assert classification.complexity in QueryComplexity
            assert 0 <= classification.confidence <= 1
    
    def test_classification_consistency(self):
        """Test that classification is consistent for the same query."""
        query = "What are our key strategic priorities for next quarter?"
        
        classifications = [self.classifier.classify(query) for _ in range(5)]
        
        # All classifications should be the same
        complexities = [c.complexity for c in classifications]
        assert len(set(complexities)) == 1  # All same complexity
        
        strategies = [c.suggested_strategy for c in classifications]
        assert len(set(strategies)) == 1  # All same strategy
    
    def test_reasoning_provided(self):
        """Test that reasoning is provided for classifications."""
        query = "Compare our market share to competitors and recommend strategic actions"
        
        classification = self.classifier.classify(query)
        
        assert isinstance(classification.reasoning, list)
        assert len(classification.reasoning) > 0
        assert all(isinstance(reason, str) for reason in classification.reasoning)
    
    def test_token_estimation(self):
        """Test token count estimation."""
        test_cases = [
            ("Short query", 2),
            ("This is a longer query with more words", 9),
            ("Very long query with many words that should result in higher token estimate", 13)
        ]
        
        for query, expected_word_count in test_cases:
            classification = self.classifier.classify(query)
            # Token estimate should be roughly word count * 1.3
            expected_tokens = expected_word_count * 1.3
            assert abs(classification.estimated_tokens - expected_tokens) < 5
    
    def test_get_classification_stats(self):
        """Test classification statistics."""
        queries = [
            "What is AI?",  # Simple
            "Explain our strategy",  # Medium  
            "Compare Q1 to Q2 performance and analyze trends with recommendations"  # Complex
        ]
        
        stats = self.classifier.get_classification_stats(queries)
        
        assert isinstance(stats, dict)
        assert "simple" in stats
        assert "medium" in stats  
        assert "complex" in stats
        assert stats["simple"] == 1
        assert stats["medium"] == 1
        assert stats["complex"] == 1
    
    def test_keyword_detection(self):
        """Test detection of complexity-indicating keywords."""
        high_complexity_query = "Analyze strategic competitive advantages and optimize ROI"
        classification = self.classifier.classify(high_complexity_query)
        
        # Should be detected as complex due to keywords
        assert classification.complexity == QueryComplexity.COMPLEX
        assert any("complexity keywords" in reason for reason in classification.reasoning)
    
    def test_multi_sentence_complexity(self):
        """Test that multi-sentence queries increase complexity."""
        single_sentence = "What is our revenue?"
        multi_sentence = "What is our revenue? How does it compare to last year? What are the trends?"
        
        single_classification = self.classifier.classify(single_sentence)
        multi_classification = self.classifier.classify(multi_sentence)
        
        # Multi-sentence should be more complex
        complexity_order = {
            QueryComplexity.SIMPLE: 1,
            QueryComplexity.MEDIUM: 2, 
            QueryComplexity.COMPLEX: 3
        }
        
        assert complexity_order[multi_classification.complexity] >= complexity_order[single_classification.complexity]


# Integration test for performance
class TestQueryClassifierPerformance:
    """Test performance characteristics of the classifier."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.classifier = ExecutiveQueryClassifier()
    
    def test_classification_speed(self):
        """Test that classification is fast enough (sub-millisecond goal)."""
        import time
        
        queries = [
            "What is our market share?",
            "Compare Q3 performance to Q2 and analyze revenue drivers with strategic recommendations",
            "Summarize board meeting minutes from last quarter"
        ]
        
        start_time = time.time()
        
        for query in queries * 10:  # Run 30 classifications
            self.classifier.classify(query)
        
        end_time = time.time()
        avg_time_ms = ((end_time - start_time) / 30) * 1000
        
        # Should be well under 1ms per classification
        assert avg_time_ms < 10  # 10ms is very generous, should be much faster
    
    def test_memory_usage(self):
        """Test that classifier doesn't have memory leaks."""
        import gc
        import sys
        
        # Get initial memory
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        # Run many classifications
        query = "What are our strategic priorities and how should we optimize performance?"
        for _ in range(100):
            self.classifier.classify(query)
        
        # Check memory hasn't grown significantly
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # Allow some growth but not excessive
        object_growth = final_objects - initial_objects
        assert object_growth < 1000  # Reasonable threshold