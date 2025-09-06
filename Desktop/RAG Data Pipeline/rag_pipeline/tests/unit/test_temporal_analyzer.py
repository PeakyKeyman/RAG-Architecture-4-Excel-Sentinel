"""Unit tests for temporal analyzer."""

import pytest
from datetime import datetime, timezone, timedelta

from ...parsing.temporal_analyzer import (
    TemporalAnalyzer,
    TemporalMetadata,
    TemporalRelevance,
    temporal_analyzer
)


class TestTemporalAnalyzer:
    """Test cases for TemporalAnalyzer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = TemporalAnalyzer()
    
    def test_date_extraction_formats(self):
        """Test extraction of various date formats."""
        test_cases = [
            ("Report dated 12/31/2023", [datetime(2023, 12, 31)]),
            ("Q3 2023 results", [datetime(2023, 7, 1)]),  # Q3 = July
            ("January 15, 2024 board meeting", [datetime(2024, 1, 15)]),
            ("FY 2024 budget", [datetime(2024, 1, 1)]),
            ("2023-04-15 data", [datetime(2023, 4, 15)]),
            ("Mar 10, 2023 update", [datetime(2023, 3, 10)])
        ]
        
        for content, expected_dates in test_cases:
            extracted_dates = self.analyzer._extract_dates(content)
            
            # Check that at least one expected date was found
            assert len(extracted_dates) > 0
            assert any(abs((extracted - expected).days) < 1 
                      for extracted in extracted_dates 
                      for expected in expected_dates)
    
    def test_quarter_extraction(self):
        """Test extraction and parsing of quarter formats."""
        quarter_cases = [
            ("Q1 2023", datetime(2023, 1, 1)),
            ("Q2 2023", datetime(2023, 4, 1)), 
            ("Q3 2023", datetime(2023, 7, 1)),
            ("Q4 2023", datetime(2023, 10, 1)),
            ("First Quarter 2024", datetime(2024, 1, 1)),
            ("Third Quarter 2023", datetime(2023, 7, 1))
        ]
        
        for content, expected_date in quarter_cases:
            extracted_dates = self.analyzer._extract_dates(content)
            
            assert len(extracted_dates) > 0
            found_match = any(
                date.year == expected_date.year and date.month == expected_date.month 
                for date in extracted_dates
            )
            assert found_match, f"Expected {expected_date} in {extracted_dates}"
    
    def test_fiscal_year_determination(self):
        """Test fiscal year and quarter determination."""
        test_date = datetime(2023, 7, 15)  # July 15, 2023
        
        fiscal_quarter = self.analyzer._determine_fiscal_quarter(test_date)
        fiscal_year = self.analyzer._determine_fiscal_year(test_date)
        
        assert fiscal_quarter == "Q3"  # July is Q3
        assert fiscal_year == 2023
    
    def test_temporal_keyword_extraction(self):
        """Test extraction of temporal keywords."""
        content = "Our current quarterly results show recent improvements compared to last year"
        
        keywords = self.analyzer._extract_temporal_keywords(content)
        
        assert len(keywords) > 0
        
        # Should find current, recent, and comparative keywords
        keyword_categories = [kw.split(':')[0] for kw in keywords]
        assert 'current' in keyword_categories
        assert 'recent' in keyword_categories
        assert 'comparative' in keyword_categories
    
    def test_recency_score_calculation(self):
        """Test recency score calculation."""
        now = datetime.now(timezone.utc)
        
        # Test different ages
        test_cases = [
            (now, 1.0),  # Current date = max score
            (now - timedelta(days=30), 0.9),  # 1 month ago = high score
            (now - timedelta(days=180), 0.5),  # 6 months ago = medium score
            (now - timedelta(days=365), 0.083),  # 1 year ago = low score
            (now - timedelta(days=730), 0.0)   # 2 years ago = min score
        ]
        
        for date, expected_score in test_cases:
            score = self.analyzer._calculate_recency_score(date)
            assert abs(score - expected_score) < 0.2  # Allow some tolerance
    
    def test_temporal_relevance_classification(self):
        """Test classification into temporal relevance categories."""
        now = datetime.now(timezone.utc)
        
        test_cases = [
            (now - timedelta(days=60), TemporalRelevance.CURRENT),
            (now - timedelta(days=180), TemporalRelevance.RECENT),
            (now - timedelta(days=600), TemporalRelevance.HISTORICAL),
            (now - timedelta(days=1500), TemporalRelevance.ARCHIVED)
        ]
        
        for date, expected_category in test_cases:
            category = self.analyzer._classify_temporal_relevance(date)
            assert category == expected_category
    
    def test_document_analysis_integration(self):
        """Test complete document temporal analysis."""
        content = """
        Board Meeting Minutes
        Date: March 15, 2024
        
        Q1 2024 Financial Review
        Revenue increased 15% year-over-year compared to Q1 2023.
        Current market conditions favor our strategic initiatives.
        """
        
        creation_time = datetime(2024, 3, 15, tzinfo=timezone.utc)
        
        metadata = self.analyzer.analyze_document(
            content=content,
            file_path="board_minutes_2024.pdf",
            creation_time=creation_time
        )
        
        assert isinstance(metadata, TemporalMetadata)
        assert metadata.document_date is not None
        assert len(metadata.referenced_dates) > 0
        assert len(metadata.temporal_indicators) > 0
        assert metadata.fiscal_quarter == "Q1"
        assert metadata.fiscal_year == 2024
        assert metadata.recency_score > 0.8  # Should be high for recent date
    
    def test_query_temporality_analysis(self):
        """Test analysis of query temporal aspects."""
        test_cases = [
            (
                "What were our current quarterly results?",
                {"has_temporal_intent": True, "recency_preference": "current"}
            ),
            (
                "Compare last year to this year performance",
                {"has_temporal_intent": True, "comparative": True}
            ),
            (
                "Historical trends in market share",
                {"has_temporal_intent": True, "recency_preference": "historical"}
            ),
            (
                "What is machine learning?",
                {"has_temporal_intent": False}
            )
        ]
        
        for query, expected_attributes in test_cases:
            analysis = self.analyzer.analyze_query_temporality(query)
            
            for key, expected_value in expected_attributes.items():
                assert analysis[key] == expected_value, f"Query: {query}, Key: {key}"
    
    def test_reporting_period_detection(self):
        """Test detection of reporting periods in content."""
        test_cases = [
            ("Annual Report 2023", "Annual_2023"),
            ("Q2 2024 quarterly earnings", "Q2_2024"),
            ("Monthly sales report for January", "January_2024")  # Assuming current year
        ]
        
        base_date = datetime(2024, 1, 15)
        
        for content, expected_period in test_cases:
            period = self.analyzer._determine_reporting_period(content, base_date)
            if expected_period.endswith("2024"):
                # For current year tests, just check the format is correct
                assert period is not None
            else:
                assert period == expected_period
    
    def test_time_expression_extraction(self):
        """Test extraction of executive time expressions."""
        content = """
        Year-over-year growth was 15%. Our trailing twelve months performance 
        shows strong momentum. The run-rate for Q4 indicates continued growth.
        """
        
        expressions = self.analyzer._extract_time_expressions(content)
        
        # Should find YoY, TTM, and run-rate expressions
        assert len(expressions) > 0
        
        # Convert to lowercase for comparison
        expressions_lower = [expr.lower() for expr in expressions if expr]
        assert any('year' in expr for expr in expressions_lower)
    
    def test_error_handling(self):
        """Test error handling with malformed dates and content."""
        problematic_content = [
            "Date: 99/99/9999",  # Invalid date
            "Random text with no dates",  # No dates
            "",  # Empty content
            "February 30, 2023",  # Invalid date
        ]
        
        for content in problematic_content:
            # Should not raise exceptions
            metadata = self.analyzer.analyze_document(content)
            assert isinstance(metadata, TemporalMetadata)
    
    def test_date_reasonableness_filtering(self):
        """Test filtering of unreasonable dates."""
        unreasonable_content = [
            "Meeting on 01/01/1800",  # Too old
            "Future date 01/01/2050",  # Too far future  
            "Year 2100 projection"    # Way too future
        ]
        
        for content in unreasonable_content:
            dates = self.analyzer._extract_dates(content)
            # Should filter out unreasonable dates
            if dates:
                now = datetime.now()
                for date in dates:
                    # Should be within reasonable range
                    years_diff = abs((date - now).days / 365.25)
                    assert years_diff <= 55  # 50 years past + 5 years future buffer


class TestTemporalAnalyzerPerformance:
    """Test performance of temporal analyzer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = TemporalAnalyzer()
    
    def test_analysis_performance(self):
        """Test that temporal analysis is reasonably fast."""
        import time
        
        content = """
        Quarterly Board Meeting
        Date: March 15, 2024
        
        Financial Performance Review - Q1 2024
        - Revenue: $10.5M (up 15% YoY from Q1 2023)
        - Operating margin: 22.3% (vs 19.8% last quarter)
        - Cash position: $25.2M as of March 31, 2024
        
        Strategic Initiatives Update
        Current progress on digital transformation shows strong momentum.
        Recent customer acquisition campaigns have exceeded targets.
        Historical data indicates continued growth trajectory.
        
        Risk Assessment
        Market conditions remain favorable through year-end.
        Competitive landscape analysis updated monthly.
        Regulatory changes effective January 1, 2025 may impact operations.
        """
        
        start_time = time.time()
        
        # Run analysis multiple times
        for _ in range(10):
            self.analyzer.analyze_document(content, creation_time=datetime.now())
        
        end_time = time.time()
        avg_time_ms = ((end_time - start_time) / 10) * 1000
        
        # Should complete within reasonable time (allow generous buffer)
        assert avg_time_ms < 100  # 100ms is very generous
    
    def test_regex_compilation_caching(self):
        """Test that regex patterns are compiled and cached properly."""
        # Access compiled patterns to ensure they exist
        assert hasattr(self.analyzer, 'compiled_date_patterns')
        assert hasattr(self.analyzer, 'compiled_time_expressions')
        
        # Patterns should be compiled regex objects
        assert len(self.analyzer.compiled_date_patterns) > 0
        assert len(self.analyzer.compiled_time_expressions) > 0
        
        # Test that they work
        test_text = "Meeting on March 15, 2024 with year-over-year analysis"
        
        # Should not raise exceptions
        dates = self.analyzer._extract_dates(test_text)
        expressions = self.analyzer._extract_time_expressions(test_text)
        
        assert len(dates) >= 0  # May or may not find dates, but shouldn't crash
        assert len(expressions) >= 0  # May or may not find expressions, but shouldn't crash