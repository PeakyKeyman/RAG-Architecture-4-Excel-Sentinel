"""Performance monitoring system for adaptive RAG pipeline."""

import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import threading
import json
import logging

from ..core.safe_logging import get_logger


@dataclass
class PerformanceMetric:
    """Individual performance metric."""
    timestamp: datetime
    metric_name: str
    value: float
    labels: Dict[str, str]
    unit: str = "ms"


@dataclass
class QueryPerformance:
    """Performance data for a single query."""
    query_id: str
    strategy: str
    complexity: str
    execution_time_ms: float
    retrieval_time_ms: Optional[float] = None
    rerank_time_ms: Optional[float] = None
    temporal_scoring_time_ms: Optional[float] = None
    total_chunks_retrieved: int = 0
    final_chunks_returned: int = 0
    success: bool = True
    error: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


class PerformanceMonitor:
    """Real-time performance monitoring for the RAG pipeline."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = get_logger(__name__)
        
        # Thread-safe storage
        self._lock = threading.RLock()
        self._metrics: deque = deque(maxlen=10000)  # Last 10k metrics
        self._query_history: deque = deque(maxlen=1000)  # Last 1k queries
        
        # Performance targets from config
        self.targets = {
            'simple_query_max_ms': self.config.get('simple_query_max_ms', 500),
            'medium_query_max_ms': self.config.get('medium_query_max_ms', 2000),
            'complex_query_max_ms': self.config.get('complex_query_max_ms', 8000),
            'retrieval_max_ms': self.config.get('retrieval_max_ms', 3000),
            'rerank_max_ms': self.config.get('rerank_max_ms', 1000),
        }
        
        # Aggregated statistics
        self._stats_cache = {}
        self._stats_cache_time = None
        self._stats_cache_ttl = 60  # 60 seconds
        
        # Alert thresholds
        self.alert_thresholds = {
            'error_rate_threshold': self.config.get('error_rate_threshold', 0.05),  # 5%
            'slow_query_threshold': self.config.get('slow_query_threshold', 0.1),   # 10%
            'p95_target_multiplier': self.config.get('p95_target_multiplier', 2.0)  # 2x target
        }
        
        # Performance alerts
        self._alerts: List[Dict] = []
        self._last_alert_check = datetime.utcnow()
        
        self.logger.info("Performance monitor initialized", extra={
            'targets': self.targets,
            'alert_thresholds': self.alert_thresholds
        })
    
    def record_query_performance(self, performance: QueryPerformance):
        """Record performance data for a query execution."""
        with self._lock:
            self._query_history.append(performance)
            
            # Record individual metrics
            labels = {
                'strategy': performance.strategy,
                'complexity': performance.complexity,
                'success': str(performance.success)
            }
            
            self._record_metric('query_execution_time', performance.execution_time_ms, labels)
            self._record_metric('chunks_retrieved', performance.total_chunks_retrieved, labels, 'count')
            self._record_metric('chunks_returned', performance.final_chunks_returned, labels, 'count')
            
            if performance.retrieval_time_ms:
                self._record_metric('retrieval_time', performance.retrieval_time_ms, labels)
            if performance.rerank_time_ms:
                self._record_metric('rerank_time', performance.rerank_time_ms, labels)
            if performance.temporal_scoring_time_ms:
                self._record_metric('temporal_scoring_time', performance.temporal_scoring_time_ms, labels)
        
        # Check for performance issues
        self._check_performance_alerts(performance)
        
        # Clear stats cache
        self._stats_cache_time = None
    
    def _record_metric(self, name: str, value: float, labels: Dict[str, str], unit: str = "ms"):
        """Record a single metric."""
        metric = PerformanceMetric(
            timestamp=datetime.utcnow(),
            metric_name=name,
            value=value,
            labels=labels,
            unit=unit
        )
        self._metrics.append(metric)
    
    def _check_performance_alerts(self, performance: QueryPerformance):
        """Check if performance violates any thresholds."""
        alerts = []
        
        # Check execution time against targets
        target_key = f"{performance.complexity.lower()}_query_max_ms"
        if target_key in self.targets:
            target_ms = self.targets[target_key]
            if performance.execution_time_ms > target_ms:
                alerts.append({
                    'type': 'slow_query',
                    'severity': 'warning' if performance.execution_time_ms < target_ms * 2 else 'critical',
                    'message': f"{performance.complexity} query took {performance.execution_time_ms:.1f}ms (target: {target_ms}ms)",
                    'query_id': performance.query_id,
                    'timestamp': datetime.utcnow()
                })
        
        # Check for errors
        if not performance.success:
            alerts.append({
                'type': 'query_error',
                'severity': 'error',
                'message': f"Query failed: {performance.error}",
                'query_id': performance.query_id,
                'timestamp': datetime.utcnow()
            })
        
        # Store alerts
        if alerts:
            with self._lock:
                self._alerts.extend(alerts)
                # Keep only last 100 alerts
                if len(self._alerts) > 100:
                    self._alerts = self._alerts[-100:]
            
            # Log critical alerts
            for alert in alerts:
                if alert['severity'] in ['critical', 'error']:
                    self.logger.error("Performance alert", extra={
                        'alert_type': alert['type'],
                        'severity': alert['severity'],
                        'message': alert['message'],
                        'query_id': alert.get('query_id')
                    })
    
    def get_performance_stats(self, window_minutes: int = 60) -> Dict[str, Any]:
        """Get aggregated performance statistics."""
        # Check cache
        now = datetime.utcnow()
        if (self._stats_cache_time and 
            (now - self._stats_cache_time).seconds < self._stats_cache_ttl):
            return self._stats_cache.copy()
        
        with self._lock:
            cutoff_time = now - timedelta(minutes=window_minutes)
            recent_queries = [
                q for q in self._query_history 
                if q.timestamp >= cutoff_time
            ]
        
        if not recent_queries:
            return {'error': 'No recent queries found'}
        
        stats = self._calculate_performance_stats(recent_queries)
        
        # Cache results
        self._stats_cache = stats
        self._stats_cache_time = now
        
        return stats.copy()
    
    def _calculate_performance_stats(self, queries: List[QueryPerformance]) -> Dict[str, Any]:
        """Calculate detailed performance statistics."""
        if not queries:
            return {}
        
        total_queries = len(queries)
        successful_queries = [q for q in queries if q.success]
        success_rate = len(successful_queries) / total_queries
        
        # Execution time statistics
        exec_times = [q.execution_time_ms for q in successful_queries]
        exec_times.sort()
        
        stats = {
            'summary': {
                'total_queries': total_queries,
                'successful_queries': len(successful_queries),
                'success_rate': round(success_rate, 4),
                'error_rate': round(1 - success_rate, 4),
                'window_minutes': (queries[-1].timestamp - queries[0].timestamp).seconds // 60
            },
            'execution_time': self._calculate_time_stats(exec_times),
            'by_complexity': {},
            'by_strategy': {},
            'targets_met': {},
            'recent_errors': []
        }
        
        # Stats by complexity
        for complexity in ['simple', 'medium', 'complex']:
            complexity_queries = [q for q in successful_queries if q.complexity.lower() == complexity]
            if complexity_queries:
                times = [q.execution_time_ms for q in complexity_queries]
                stats['by_complexity'][complexity] = {
                    'count': len(complexity_queries),
                    **self._calculate_time_stats(times)
                }
                
                # Check target adherence
                target_key = f"{complexity}_query_max_ms"
                if target_key in self.targets:
                    target = self.targets[target_key]
                    within_target = sum(1 for t in times if t <= target)
                    stats['targets_met'][complexity] = {
                        'target_ms': target,
                        'within_target_count': within_target,
                        'within_target_percentage': round(within_target / len(times) * 100, 1)
                    }
        
        # Stats by strategy
        for strategy in ['direct_vector', 'hybrid_rerank', 'full_ensemble']:
            strategy_queries = [q for q in successful_queries if q.strategy == strategy]
            if strategy_queries:
                times = [q.execution_time_ms for q in strategy_queries]
                stats['by_strategy'][strategy] = {
                    'count': len(strategy_queries),
                    **self._calculate_time_stats(times)
                }
        
        # Recent errors
        error_queries = [q for q in queries if not q.success]
        stats['recent_errors'] = [
            {
                'query_id': q.query_id,
                'error': q.error,
                'timestamp': q.timestamp.isoformat(),
                'strategy': q.strategy,
                'complexity': q.complexity
            }
            for q in error_queries[-10:]  # Last 10 errors
        ]
        
        return stats
    
    def _calculate_time_stats(self, times: List[float]) -> Dict[str, float]:
        """Calculate time statistics for a list of execution times."""
        if not times:
            return {}
        
        times.sort()
        n = len(times)
        
        return {
            'mean': round(sum(times) / n, 2),
            'median': round(times[n // 2], 2),
            'p95': round(times[int(n * 0.95)], 2) if n >= 20 else round(times[-1], 2),
            'p99': round(times[int(n * 0.99)], 2) if n >= 100 else round(times[-1], 2),
            'min': round(min(times), 2),
            'max': round(max(times), 2)
        }
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get current active performance alerts."""
        with self._lock:
            return self._alerts.copy()
    
    def clear_alerts(self):
        """Clear all active alerts."""
        with self._lock:
            self._alerts.clear()
        self.logger.info("Performance alerts cleared")
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on system performance."""
        recent_stats = self.get_performance_stats(window_minutes=15)
        
        if 'error' in recent_stats:
            return {
                'status': 'unknown',
                'message': recent_stats['error'],
                'timestamp': datetime.utcnow().isoformat()
            }
        
        summary = recent_stats.get('summary', {})
        error_rate = summary.get('error_rate', 0)
        
        # Determine health status
        if error_rate > self.alert_thresholds['error_rate_threshold']:
            status = 'unhealthy'
            message = f"High error rate: {error_rate:.2%}"
        elif len(self._alerts) > 10:
            status = 'degraded'
            message = f"Multiple performance alerts: {len(self._alerts)}"
        else:
            status = 'healthy'
            message = "All systems operating within parameters"
        
        return {
            'status': status,
            'message': message,
            'error_rate': error_rate,
            'total_queries_15min': summary.get('total_queries', 0),
            'active_alerts': len(self._alerts),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def export_metrics(self, format: str = 'json') -> str:
        """Export performance metrics in specified format."""
        stats = self.get_performance_stats()
        
        if format.lower() == 'json':
            return json.dumps(stats, indent=2, default=str)
        elif format.lower() == 'prometheus':
            return self._export_prometheus_format(stats)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_prometheus_format(self, stats: Dict) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        summary = stats.get('summary', {})
        
        # Basic metrics
        lines.append(f"rag_queries_total {summary.get('total_queries', 0)}")
        lines.append(f"rag_queries_success_rate {summary.get('success_rate', 0)}")
        lines.append(f"rag_queries_error_rate {summary.get('error_rate', 0)}")
        
        # Execution time metrics
        exec_time = stats.get('execution_time', {})
        for metric, value in exec_time.items():
            lines.append(f"rag_execution_time_{metric} {value}")
        
        # By complexity
        for complexity, data in stats.get('by_complexity', {}).items():
            for metric, value in data.items():
                if isinstance(value, (int, float)):
                    lines.append(f'rag_complexity_{metric}{{complexity="{complexity}"}} {value}')
        
        return '\n'.join(lines)


# Global performance monitor instance
_monitor_instance: Optional[PerformanceMonitor] = None
_monitor_lock = threading.Lock()


def get_performance_monitor(config: Optional[Dict] = None) -> PerformanceMonitor:
    """Get or create the global performance monitor instance."""
    global _monitor_instance
    
    with _monitor_lock:
        if _monitor_instance is None:
            _monitor_instance = PerformanceMonitor(config)
        return _monitor_instance


def record_query_performance(query_id: str, strategy: str, complexity: str, 
                           execution_time_ms: float, **kwargs):
    """Convenience function to record query performance."""
    performance = QueryPerformance(
        query_id=query_id,
        strategy=strategy,
        complexity=complexity,
        execution_time_ms=execution_time_ms,
        **kwargs
    )
    
    monitor = get_performance_monitor()
    monitor.record_query_performance(performance)


class PerformanceTimer:
    """Context manager for timing operations."""
    
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
    
    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        if self.start_time is None or self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time) * 1000