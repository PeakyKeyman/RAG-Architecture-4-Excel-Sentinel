"""
Time-aware relevance scoring for search results.

Adjusts document relevance scores based on temporal factors:
- Document recency for different query types
- Temporal query intent matching
- Business context temporal weighting
- Strategic vs operational document prioritization
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass
from enum import Enum

from ..core.logging_config import get_logger
from ..parsing.temporal_analyzer import temporal_analyzer, TemporalRelevance


logger = get_logger(__name__, "temporal_scorer")


class TemporalScoringStrategy(Enum):
    """Different temporal scoring strategies."""
    RECENCY_BOOST = "recency_boost"           # Boost recent documents
    RELEVANCE_MATCH = "relevance_match"       # Match query temporal intent
    BUSINESS_CONTEXT = "business_context"     # Weight by business importance
    BALANCED = "balanced"                     # Combine multiple factors


@dataclass
class TemporalScoringConfig:
    """Configuration for temporal scoring."""
    
    # Strategy selection
    strategy: TemporalScoringStrategy = TemporalScoringStrategy.BALANCED
    
    # Recency boost parameters
    recency_boost_factor: float = 0.3         # How much to boost recent docs (0-1)
    current_boost: float = 1.5                # Multiplier for current docs
    recent_boost: float = 1.2                 # Multiplier for recent docs
    historical_penalty: float = 0.8           # Multiplier for historical docs
    archived_penalty: float = 0.5             # Multiplier for archived docs
    
    # Query type weights
    strategy_recency_weight: float = 0.8      # Strategic docs prefer recency
    financial_recency_weight: float = 0.9     # Financial docs strongly prefer recency
    policy_recency_weight: float = 0.4        # Policy docs less time-sensitive
    operational_recency_weight: float = 0.6   # Operational docs moderately time-sensitive
    
    # Temporal intent matching
    temporal_intent_boost: float = 0.2        # Boost for temporal intent match
    comparative_boost: float = 0.15           # Extra boost for comparative queries
    
    # Business context
    quarterly_report_boost: float = 0.1       # Boost for quarterly reports
    annual_report_boost: float = 0.15         # Boost for annual reports
    board_meeting_boost: float = 0.1          # Boost for board documents


class TemporalScorer:
    """
    Time-aware relevance scorer for executive documents.
    
    Applies temporal adjustments to search result scores based on:
    - Document age and relevance category
    - Query temporal intent
    - Business document types
    - Seasonal/periodic relevance
    """
    
    def __init__(self, config: Optional[TemporalScoringConfig] = None):
        """Initialize temporal scorer with configuration."""
        self.config = config or TemporalScoringConfig()
        self.logger = get_logger(__name__, "temporal_scorer")
    
    def score_results(self, 
                     query: str,
                     results: List[Dict[str, Any]],
                     strategy: Optional[TemporalScoringStrategy] = None) -> List[Dict[str, Any]]:
        """
        Apply temporal scoring to search results.
        
        Args:
            query: Original search query
            results: List of search result dictionaries with metadata
            strategy: Optional strategy override
            
        Returns:
            List of results with updated temporal_score and combined_score
        """
        if not results:
            return results
        
        # Analyze query for temporal intent
        query_temporal_info = temporal_analyzer.analyze_query_temporality(query)
        
        # Use provided strategy or config default
        scoring_strategy = strategy or self.config.strategy
        
        scored_results = []
        
        for result in results:
            try:
                # Extract metadata
                metadata = result.get('metadata', {})
                original_score = result.get('score', 0.0)
                
                # Calculate temporal score components
                temporal_scores = self._calculate_temporal_scores(
                    result=result,
                    query_temporal_info=query_temporal_info,
                    strategy=scoring_strategy
                )
                
                # Apply scoring strategy
                final_temporal_score = self._apply_scoring_strategy(
                    temporal_scores=temporal_scores,
                    strategy=scoring_strategy,
                    document_type=metadata.get('document_type', 'general')
                )
                
                # Combine with original score
                combined_score = self._combine_scores(
                    original_score=original_score,
                    temporal_score=final_temporal_score,
                    query_temporal_info=query_temporal_info
                )
                
                # Create enhanced result
                enhanced_result = result.copy()
                enhanced_result['temporal_score'] = final_temporal_score
                enhanced_result['combined_score'] = combined_score
                enhanced_result['temporal_scores_breakdown'] = temporal_scores
                enhanced_result['query_temporal_match'] = query_temporal_info['has_temporal_intent']
                
                scored_results.append(enhanced_result)
                
            except Exception as e:
                # On error, keep original result
                self.logger.warning(f"Temporal scoring failed for result: {str(e)}")
                enhanced_result = result.copy()
                enhanced_result['temporal_score'] = 1.0  # Neutral score
                enhanced_result['combined_score'] = result.get('score', 0.0)
                scored_results.append(enhanced_result)
        
        # Sort by combined score
        scored_results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        self.logger.debug(
            f"Temporal scoring applied to {len(results)} results",
            extra={
                "strategy": scoring_strategy.value,
                "query_has_temporal_intent": query_temporal_info['has_temporal_intent'],
                "avg_temporal_score": sum(r['temporal_score'] for r in scored_results) / len(scored_results)
            }
        )
        
        return scored_results
    
    def _calculate_temporal_scores(self,
                                  result: Dict[str, Any],
                                  query_temporal_info: Dict[str, Any],
                                  strategy: TemporalScoringStrategy) -> Dict[str, float]:
        """Calculate individual temporal score components."""
        metadata = result.get('metadata', {})
        scores = {}
        
        # 1. Recency Score (based on document age)
        recency_score = metadata.get('recency_score', 0.5)
        temporal_relevance = metadata.get('temporal_relevance', 'historical')
        
        if temporal_relevance == 'current':
            scores['recency'] = recency_score * self.config.current_boost
        elif temporal_relevance == 'recent':
            scores['recency'] = recency_score * self.config.recent_boost
        elif temporal_relevance == 'historical':
            scores['recency'] = recency_score * self.config.historical_penalty
        else:  # archived
            scores['recency'] = recency_score * self.config.archived_penalty
        
        # 2. Query Intent Matching
        scores['intent_match'] = self._calculate_intent_match_score(
            query_temporal_info=query_temporal_info,
            document_temporal_data=metadata
        )
        
        # 3. Business Context Score
        scores['business_context'] = self._calculate_business_context_score(metadata)
        
        # 4. Document Type Temporal Weight
        document_type = metadata.get('document_type', 'general')
        scores['doc_type_weight'] = self._get_document_type_temporal_weight(document_type)
        
        return scores
    
    def _calculate_intent_match_score(self,
                                    query_temporal_info: Dict[str, Any],
                                    document_temporal_data: Dict[str, Any]) -> float:
        """Calculate how well document matches query temporal intent."""
        if not query_temporal_info.get('has_temporal_intent', False):
            return 1.0  # Neutral if no temporal intent
        
        recency_preference = query_temporal_info.get('recency_preference', 'current')
        document_relevance = document_temporal_data.get('temporal_relevance', 'historical')
        
        # Direct matches get highest score
        if recency_preference == document_relevance:
            return 1.0 + self.config.temporal_intent_boost
        
        # Related matches get moderate boost
        if (recency_preference == 'current' and document_relevance == 'recent') or \
           (recency_preference == 'recent' and document_relevance in ['current', 'historical']):
            return 1.0 + (self.config.temporal_intent_boost * 0.5)
        
        # Comparative queries get special handling
        if query_temporal_info.get('comparative', False):
            return 1.0 + self.config.comparative_boost
        
        # Mismatches get slight penalty
        return 0.9
    
    def _calculate_business_context_score(self, metadata: Dict[str, Any]) -> float:
        """Calculate business context relevance score."""
        base_score = 1.0
        
        # Boost for specific reporting periods
        reporting_period = metadata.get('reporting_period', '')
        
        if 'quarterly' in reporting_period.lower() or reporting_period.startswith('Q'):
            base_score += self.config.quarterly_report_boost
        elif 'annual' in reporting_period.lower():
            base_score += self.config.annual_report_boost
        
        # Boost for high-importance document types
        document_type = metadata.get('document_type', 'general')
        if document_type == 'board_investor':
            base_score += self.config.board_meeting_boost
        
        # Consider fiscal alignment
        fiscal_quarter = metadata.get('fiscal_quarter')
        if fiscal_quarter:
            # Could add seasonal relevance logic here
            pass
        
        return base_score
    
    def _get_document_type_temporal_weight(self, document_type: str) -> float:
        """Get temporal weighting factor for document type."""
        type_weights = {
            'financial': self.config.financial_recency_weight,
            'strategic': self.config.strategy_recency_weight,
            'board_investor': self.config.strategy_recency_weight,
            'policy_compliance': self.config.policy_recency_weight,
            'market_research': self.config.operational_recency_weight,
            'general': 0.5
        }
        
        return type_weights.get(document_type, 0.5)
    
    def _apply_scoring_strategy(self,
                              temporal_scores: Dict[str, float],
                              strategy: TemporalScoringStrategy,
                              document_type: str) -> float:
        """Apply the selected temporal scoring strategy."""
        
        if strategy == TemporalScoringStrategy.RECENCY_BOOST:
            # Simple recency-based scoring
            return temporal_scores['recency']
        
        elif strategy == TemporalScoringStrategy.RELEVANCE_MATCH:
            # Focus on query intent matching
            return (temporal_scores['recency'] * 0.3 + 
                   temporal_scores['intent_match'] * 0.7)
        
        elif strategy == TemporalScoringStrategy.BUSINESS_CONTEXT:
            # Weight by business importance
            return (temporal_scores['recency'] * 0.4 + 
                   temporal_scores['business_context'] * 0.6)
        
        else:  # BALANCED
            # Combine all factors with document-type specific weighting
            doc_type_weight = temporal_scores['doc_type_weight']
            
            weighted_score = (
                temporal_scores['recency'] * 0.3 * doc_type_weight +
                temporal_scores['intent_match'] * 0.4 +
                temporal_scores['business_context'] * 0.3
            )
            
            return weighted_score
    
    def _combine_scores(self,
                       original_score: float,
                       temporal_score: float,
                       query_temporal_info: Dict[str, Any]) -> float:
        """Combine original relevance score with temporal score."""
        
        # Determine temporal weight based on query
        if query_temporal_info.get('has_temporal_intent', False):
            # Temporal queries get higher temporal weighting
            temporal_weight = 0.4
        else:
            # Non-temporal queries get lower temporal weighting
            temporal_weight = 0.2
        
        # Weighted combination
        combined_score = (
            original_score * (1 - temporal_weight) +
            original_score * temporal_score * temporal_weight
        )
        
        return combined_score
    
    def get_temporal_boost_explanation(self, result: Dict[str, Any]) -> str:
        """Get human-readable explanation of temporal boost applied."""
        temporal_scores = result.get('temporal_scores_breakdown', {})
        metadata = result.get('metadata', {})
        
        explanations = []
        
        # Recency explanation
        temporal_relevance = metadata.get('temporal_relevance', 'historical')
        if temporal_relevance == 'current':
            explanations.append("Recent document boost")
        elif temporal_relevance == 'archived':
            explanations.append("Archived document penalty")
        
        # Intent matching
        if result.get('query_temporal_match', False):
            explanations.append("Temporal intent match")
        
        # Business context
        reporting_period = metadata.get('reporting_period', '')
        if reporting_period:
            explanations.append(f"Reporting period relevance ({reporting_period})")
        
        return "; ".join(explanations) if explanations else "No temporal adjustment"


# Global temporal scorer instance
temporal_scorer = TemporalScorer()