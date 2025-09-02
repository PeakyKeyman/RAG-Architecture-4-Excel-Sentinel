"""Knowledge base adjustment engine for feedback-driven improvements."""

import time
import json
import threading
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from collections import defaultdict, Counter
from enum import Enum

from ..core.config import settings
from ..core.exceptions import KnowledgeBaseException
from ..core.logging_config import get_logger, log_performance


class FeedbackType(Enum):
    """Types of feedback for knowledge base adjustment."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    IRRELEVANT = "irrelevant"
    OUTDATED = "outdated"
    INCORRECT = "incorrect"


class AdjustmentAction(Enum):
    """Types of adjustment actions."""
    DOWNWEIGHT = "downweight"
    UPWEIGHT = "upweight"
    REMOVE = "remove"
    UPDATE_METADATA = "update_metadata"
    QUARANTINE = "quarantine"


@dataclass
class FeedbackEvent:
    """Container for user feedback events."""
    event_id: str
    chunk_id: str
    parent_id: Optional[str]
    document_id: str
    query: str
    feedback_type: FeedbackType
    relevance_score: float
    user_rating: Optional[float]
    comment: Optional[str]
    timestamp: datetime
    context_data: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['feedback_type'] = self.feedback_type.value
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class ChunkAnalytics:
    """Analytics for chunk performance."""
    chunk_id: str
    parent_id: Optional[str]
    document_id: str
    total_retrievals: int
    positive_feedback: int
    negative_feedback: int
    avg_relevance_score: float
    avg_user_rating: float
    user_ratings_count: int  # Count of user ratings for proper averaging
    feedback_ratio: float  # positive / total feedback
    last_retrieved: datetime
    created_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['last_retrieved'] = self.last_retrieved.isoformat()
        data['created_at'] = self.created_at.isoformat()
        return data


@dataclass
class AdjustmentRecommendation:
    """Recommendation for knowledge base adjustment."""
    chunk_id: str
    action: AdjustmentAction
    confidence: float
    reason: str
    analytics: ChunkAnalytics
    supporting_evidence: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['action'] = self.action.value
        data['analytics'] = self.analytics.to_dict()
        return data


class KnowledgeBaseAdjustmentEngine:
    """Engine for adjusting knowledge base based on feedback and analytics."""
    
    def __init__(self, min_feedback_count: int = 5, confidence_threshold: float = 0.7):
        self.logger = get_logger(__name__, "kb_adjustment")
        self.min_feedback_count = min_feedback_count
        self.confidence_threshold = confidence_threshold
        
        # In-memory storage (would be replaced with persistent storage)
        self.feedback_events: List[FeedbackEvent] = []
        self.chunk_analytics: Dict[str, ChunkAnalytics] = {}
        self.adjustment_history: List[Dict[str, Any]] = []
        
        # Thread safety locks
        self._global_lock = threading.RLock()  # For feedback_events list
        self._chunk_locks: Dict[str, threading.RLock] = defaultdict(threading.RLock)  # Per-chunk analytics locks
    
    def record_feedback(
        self,
        chunk_id: str,
        query: str,
        feedback_type: FeedbackType,
        relevance_score: float,
        parent_id: Optional[str] = None,
        document_id: Optional[str] = None,
        user_rating: Optional[float] = None,
        comment: Optional[str] = None,
        context_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """Record user feedback for a chunk."""
        try:
            event_id = f"feedback_{int(time.time())}_{chunk_id[:8]}"
            
            feedback_event = FeedbackEvent(
                event_id=event_id,
                chunk_id=chunk_id,
                parent_id=parent_id,
                document_id=document_id or "unknown",
                query=query,
                feedback_type=feedback_type,
                relevance_score=relevance_score,
                user_rating=user_rating,
                comment=comment,
                timestamp=datetime.now(timezone.utc),
                context_data=context_data or {}
            )
            
            # Thread-safe append to feedback events
            with self._global_lock:
                self.feedback_events.append(feedback_event)
            
            # Update chunk analytics
            self._update_chunk_analytics(feedback_event)
            
            self.logger.info(
                f"Recorded {feedback_type.value} feedback for chunk {chunk_id}",
                extra={
                    "event_id": event_id,
                    "chunk_id": chunk_id,
                    "feedback_type": feedback_type.value,
                    "relevance_score": relevance_score
                }
            )
            
            return event_id
            
        except Exception as e:
            raise KnowledgeBaseException(
                f"Failed to record feedback: {str(e)}",
                component="kb_adjustment",
                error_code="FEEDBACK_RECORDING_FAILED",
                details={
                    "chunk_id": chunk_id,
                    "feedback_type": feedback_type.value
                }
            )
    
    def _get_chunk_lock(self, chunk_id: str) -> threading.RLock:
        """Get or create a lock for a specific chunk."""
        with self._global_lock:
            return self._chunk_locks[chunk_id]
    
    def _update_chunk_analytics(self, feedback_event: FeedbackEvent) -> None:
        """Update analytics for a chunk based on feedback with thread safety."""
        chunk_id = feedback_event.chunk_id
        chunk_lock = self._get_chunk_lock(chunk_id)
        
        with chunk_lock:
            if chunk_id not in self.chunk_analytics:
                self.chunk_analytics[chunk_id] = ChunkAnalytics(
                    chunk_id=chunk_id,
                    parent_id=feedback_event.parent_id,
                    document_id=feedback_event.document_id,
                    total_retrievals=1,
                    positive_feedback=0,
                    negative_feedback=0,
                    avg_relevance_score=feedback_event.relevance_score,
                    avg_user_rating=feedback_event.user_rating or 0.0,
                    user_ratings_count=1 if feedback_event.user_rating is not None else 0,
                    feedback_ratio=0.0,
                    last_retrieved=feedback_event.timestamp,
                    created_at=feedback_event.timestamp
                )
            
            analytics = self.chunk_analytics[chunk_id]
            
            # Update counters
            analytics.total_retrievals += 1
            analytics.last_retrieved = feedback_event.timestamp
            
            # Update feedback counts
            if feedback_event.feedback_type in [FeedbackType.POSITIVE]:
                analytics.positive_feedback += 1
            elif feedback_event.feedback_type in [FeedbackType.NEGATIVE, FeedbackType.IRRELEVANT, FeedbackType.INCORRECT]:
                analytics.negative_feedback += 1
            
            # Update running averages
            total_feedback = analytics.positive_feedback + analytics.negative_feedback
            if total_feedback > 0:
                analytics.feedback_ratio = analytics.positive_feedback / total_feedback
            
            # Update relevance score (running average)
            analytics.avg_relevance_score = (
                (analytics.avg_relevance_score * (analytics.total_retrievals - 1) + 
                 feedback_event.relevance_score) / analytics.total_retrievals
            )
            
            # Update user rating (if provided) - proper running average
            if feedback_event.user_rating is not None:
                # Calculate proper running average
                total_rating = analytics.avg_user_rating * analytics.user_ratings_count
                analytics.user_ratings_count += 1
                analytics.avg_user_rating = (total_rating + feedback_event.user_rating) / analytics.user_ratings_count
    
    def analyze_knowledge_base(self, min_retrievals: int = None) -> List[AdjustmentRecommendation]:
        """Analyze knowledge base and generate adjustment recommendations."""
        min_retrievals = min_retrievals or self.min_feedback_count
        
        try:
            start_time = time.time()
            
            recommendations = []
            analyzed_chunks = 0
            
            for chunk_id, analytics in self.chunk_analytics.items():
                if analytics.total_retrievals < min_retrievals:
                    continue
                
                analyzed_chunks += 1
                
                # Generate recommendations based on analytics
                chunk_recommendations = self._analyze_chunk_performance(analytics)
                recommendations.extend(chunk_recommendations)
            
            # Sort recommendations by confidence
            recommendations.sort(key=lambda x: x.confidence, reverse=True)
            
            # Filter by confidence threshold
            high_confidence_recommendations = [
                rec for rec in recommendations 
                if rec.confidence >= self.confidence_threshold
            ]
            
            duration = (time.time() - start_time) * 1000
            
            log_performance(
                self.logger,
                "analyze_knowledge_base",
                duration,
                metadata={
                    "analyzed_chunks": analyzed_chunks,
                    "total_recommendations": len(recommendations),
                    "high_confidence_recommendations": len(high_confidence_recommendations),
                    "min_retrievals": min_retrievals
                }
            )
            
            self.logger.info(
                f"Knowledge base analysis complete: {len(high_confidence_recommendations)} recommendations",
                extra={
                    "analyzed_chunks": analyzed_chunks,
                    "recommendations": len(high_confidence_recommendations)
                }
            )
            
            return high_confidence_recommendations
            
        except Exception as e:
            raise KnowledgeBaseException(
                f"Failed to analyze knowledge base: {str(e)}",
                component="kb_adjustment",
                error_code="ANALYSIS_FAILED"
            )
    
    def _analyze_chunk_performance(self, analytics: ChunkAnalytics) -> List[AdjustmentRecommendation]:
        """Analyze individual chunk performance and generate recommendations."""
        recommendations = []
        
        # Rule 1: Low feedback ratio - downweight or remove
        if analytics.feedback_ratio < 0.3 and analytics.total_retrievals >= self.min_feedback_count:
            confidence = 1.0 - analytics.feedback_ratio  # Higher confidence for worse performance
            
            if analytics.feedback_ratio < 0.1:
                action = AdjustmentAction.REMOVE
                reason = f"Very poor feedback ratio: {analytics.feedback_ratio:.2f}"
            else:
                action = AdjustmentAction.DOWNWEIGHT
                reason = f"Poor feedback ratio: {analytics.feedback_ratio:.2f}"
            
            recommendations.append(AdjustmentRecommendation(
                chunk_id=analytics.chunk_id,
                action=action,
                confidence=confidence,
                reason=reason,
                analytics=analytics,
                supporting_evidence={
                    "feedback_ratio": analytics.feedback_ratio,
                    "negative_feedback": analytics.negative_feedback,
                    "total_retrievals": analytics.total_retrievals
                }
            ))
        
        # Rule 2: High feedback ratio - upweight
        elif analytics.feedback_ratio > 0.8 and analytics.positive_feedback >= 3:
            confidence = analytics.feedback_ratio
            
            recommendations.append(AdjustmentRecommendation(
                chunk_id=analytics.chunk_id,
                action=AdjustmentAction.UPWEIGHT,
                confidence=confidence,
                reason=f"Excellent feedback ratio: {analytics.feedback_ratio:.2f}",
                analytics=analytics,
                supporting_evidence={
                    "feedback_ratio": analytics.feedback_ratio,
                    "positive_feedback": analytics.positive_feedback
                }
            ))
        
        # Rule 3: Low relevance score - investigate
        if analytics.avg_relevance_score < 0.3:
            recommendations.append(AdjustmentRecommendation(
                chunk_id=analytics.chunk_id,
                action=AdjustmentAction.DOWNWEIGHT,
                confidence=0.6,
                reason=f"Low average relevance score: {analytics.avg_relevance_score:.2f}",
                analytics=analytics,
                supporting_evidence={
                    "avg_relevance_score": analytics.avg_relevance_score,
                    "total_retrievals": analytics.total_retrievals
                }
            ))
        
        # Rule 4: Outdated content (would need last_updated field)
        days_since_created = (datetime.now(timezone.utc) - analytics.created_at).days
        if days_since_created > 365:  # Content older than 1 year
            recommendations.append(AdjustmentRecommendation(
                chunk_id=analytics.chunk_id,
                action=AdjustmentAction.UPDATE_METADATA,
                confidence=0.5,
                reason=f"Content is {days_since_created} days old - may need review",
                analytics=analytics,
                supporting_evidence={
                    "days_old": days_since_created,
                    "created_at": analytics.created_at.isoformat()
                }
            ))
        
        return recommendations
    
    def apply_adjustments(
        self,
        recommendations: List[AdjustmentRecommendation],
        dry_run: bool = True
    ) -> Dict[str, Any]:
        """Apply knowledge base adjustments based on recommendations."""
        if dry_run:
            self.logger.info("Running adjustment in dry-run mode - no changes will be made")
        
        try:
            start_time = time.time()
            
            results = {
                "applied": 0,
                "skipped": 0,
                "errors": 0,
                "actions": defaultdict(int)
            }
            
            for recommendation in recommendations:
                try:
                    if dry_run:
                        self.logger.info(
                            f"[DRY RUN] Would apply {recommendation.action.value} to {recommendation.chunk_id}",
                            extra={
                                "chunk_id": recommendation.chunk_id,
                                "action": recommendation.action.value,
                                "confidence": recommendation.confidence,
                                "reason": recommendation.reason
                            }
                        )
                        results["applied"] += 1
                    else:
                        success = self._apply_single_adjustment(recommendation)
                        if success:
                            results["applied"] += 1
                        else:
                            results["skipped"] += 1
                    
                    results["actions"][recommendation.action.value] += 1
                    
                except Exception as e:
                    results["errors"] += 1
                    self.logger.error(
                        f"Failed to apply adjustment to {recommendation.chunk_id}: {str(e)}",
                        extra={
                            "chunk_id": recommendation.chunk_id,
                            "action": recommendation.action.value
                        }
                    )
            
            # Record adjustment run
            adjustment_run = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "dry_run": dry_run,
                "results": dict(results),
                "recommendations_count": len(recommendations)
            }
            self.adjustment_history.append(adjustment_run)
            
            duration = (time.time() - start_time) * 1000
            
            log_performance(
                self.logger,
                "apply_adjustments",
                duration,
                success=results["errors"] == 0,
                metadata={
                    "dry_run": dry_run,
                    "recommendations": len(recommendations),
                    **dict(results)
                }
            )
            
            return dict(results)
            
        except Exception as e:
            raise KnowledgeBaseException(
                f"Failed to apply adjustments: {str(e)}",
                component="kb_adjustment",
                error_code="ADJUSTMENT_APPLICATION_FAILED"
            )
    
    def _apply_single_adjustment(self, recommendation: AdjustmentRecommendation) -> bool:
        """Apply a single adjustment recommendation."""
        # In a real implementation, this would update the vector store
        # For now, we just log the action
        
        action_map = {
            AdjustmentAction.DOWNWEIGHT: self._downweight_chunk,
            AdjustmentAction.UPWEIGHT: self._upweight_chunk,
            AdjustmentAction.REMOVE: self._remove_chunk,
            AdjustmentAction.UPDATE_METADATA: self._update_chunk_metadata,
            AdjustmentAction.QUARANTINE: self._quarantine_chunk
        }
        
        handler = action_map.get(recommendation.action)
        if handler:
            return handler(recommendation)
        
        return False
    
    def _downweight_chunk(self, recommendation: AdjustmentRecommendation) -> bool:
        """Downweight a chunk in the vector store."""
        # Implementation would update vector store weights/metadata
        self.logger.info(f"Downweighting chunk {recommendation.chunk_id}")
        return True
    
    def _upweight_chunk(self, recommendation: AdjustmentRecommendation) -> bool:
        """Upweight a chunk in the vector store."""
        # Implementation would update vector store weights/metadata
        self.logger.info(f"Upweighting chunk {recommendation.chunk_id}")
        return True
    
    def _remove_chunk(self, recommendation: AdjustmentRecommendation) -> bool:
        """Remove a chunk from the vector store."""
        # Implementation would remove from vector store
        self.logger.info(f"Removing chunk {recommendation.chunk_id}")
        return True
    
    def _update_chunk_metadata(self, recommendation: AdjustmentRecommendation) -> bool:
        """Update chunk metadata."""
        # Implementation would update metadata in vector store
        self.logger.info(f"Updating metadata for chunk {recommendation.chunk_id}")
        return True
    
    def _quarantine_chunk(self, recommendation: AdjustmentRecommendation) -> bool:
        """Quarantine a problematic chunk."""
        # Implementation would mark chunk as quarantined
        self.logger.info(f"Quarantining chunk {recommendation.chunk_id}")
        return True
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """Get summary analytics for the knowledge base."""
        if not self.chunk_analytics:
            return {"total_chunks": 0, "total_feedback": 0}
        
        total_chunks = len(self.chunk_analytics)
        total_feedback = len(self.feedback_events)
        
        feedback_ratios = [a.feedback_ratio for a in self.chunk_analytics.values()]
        relevance_scores = [a.avg_relevance_score for a in self.chunk_analytics.values()]
        
        return {
            "total_chunks": total_chunks,
            "total_feedback_events": total_feedback,
            "avg_feedback_ratio": sum(feedback_ratios) / len(feedback_ratios) if feedback_ratios else 0,
            "avg_relevance_score": sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0,
            "chunks_with_sufficient_feedback": sum(
                1 for a in self.chunk_analytics.values() 
                if a.total_retrievals >= self.min_feedback_count
            ),
            "adjustment_runs": len(self.adjustment_history)
        }


# Global knowledge base adjustment engine
kb_adjustment_engine = KnowledgeBaseAdjustmentEngine()