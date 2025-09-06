"""API endpoints for performance monitoring and feature flag management."""

from fastapi import APIRouter, HTTPException, Query, Body
from typing import Dict, Any, Optional, List
from datetime import datetime

from ..monitoring.performance_monitor import get_performance_monitor
from ..monitoring.feature_flags import get_feature_flag_manager
from ..inference.adaptive_router import adaptive_router
from ..core.safe_logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/monitoring", tags=["monitoring"])

# Initialize managers
performance_monitor = get_performance_monitor()
feature_flag_manager = get_feature_flag_manager()


@router.get("/health")
async def health_check():
    """Get overall system health status."""
    try:
        perf_health = performance_monitor.health_check()
        flag_health = feature_flag_manager.health_check()
        
        # Determine overall status
        if perf_health['status'] == 'unhealthy' or flag_health['status'] != 'healthy':
            overall_status = 'unhealthy'
        elif perf_health['status'] == 'degraded':
            overall_status = 'degraded'
        else:
            overall_status = 'healthy'
        
        return {
            'status': overall_status,
            'timestamp': datetime.utcnow().isoformat(),
            'components': {
                'performance': perf_health,
                'feature_flags': flag_health,
                'adaptive_router': {
                    'status': 'healthy',
                    'total_queries': adaptive_router.metrics.total_queries
                }
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Health check failed")


@router.get("/performance/stats")
async def get_performance_stats(
    window_minutes: int = Query(60, description="Time window in minutes"),
    format: str = Query("json", description="Response format (json|prometheus)")
):
    """Get performance statistics for specified time window."""
    try:
        if format.lower() == "prometheus":
            stats = performance_monitor.export_metrics("prometheus")
            return {"data": stats, "format": "prometheus"}
        else:
            stats = performance_monitor.get_performance_stats(window_minutes)
            return stats
    except Exception as e:
        logger.error(f"Failed to get performance stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve performance statistics")


@router.get("/performance/alerts")
async def get_performance_alerts():
    """Get current performance alerts."""
    try:
        alerts = performance_monitor.get_active_alerts()
        return {
            'alerts': alerts,
            'total_alerts': len(alerts),
            'timestamp': datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get alerts: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve alerts")


@router.post("/performance/alerts/clear")
async def clear_performance_alerts():
    """Clear all performance alerts."""
    try:
        performance_monitor.clear_alerts()
        return {
            'success': True,
            'message': 'All performance alerts cleared',
            'timestamp': datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to clear alerts: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to clear alerts")


@router.get("/feature-flags")
async def get_all_feature_flags(org_id: Optional[str] = Query(None)):
    """Get all feature flags with their current values."""
    try:
        flags = feature_flag_manager.get_all_flags(org_id)
        return {
            'feature_flags': flags,
            'total_flags': len(flags),
            'org_id': org_id,
            'timestamp': datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get feature flags: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve feature flags")


@router.get("/feature-flags/{flag_name}")
async def get_feature_flag(
    flag_name: str,
    org_id: Optional[str] = Query(None),
    user_id: Optional[str] = Query(None)
):
    """Get specific feature flag value."""
    try:
        value = feature_flag_manager.get_flag(flag_name, org_id=org_id, user_id=user_id)
        if value is None:
            raise HTTPException(status_code=404, detail=f"Feature flag '{flag_name}' not found")
        
        return {
            'flag_name': flag_name,
            'value': value,
            'org_id': org_id,
            'user_id': user_id,
            'timestamp': datetime.utcnow().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get feature flag {flag_name}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve feature flag")


@router.post("/feature-flags/{flag_name}")
async def set_feature_flag(
    flag_name: str,
    request: Dict[str, Any] = Body(...),
    org_id: Optional[str] = Query(None)
):
    """Set or update feature flag value."""
    try:
        value = request.get('value')
        description = request.get('description')
        
        if value is None:
            raise HTTPException(status_code=400, detail="Value is required")
        
        success = feature_flag_manager.set_flag(
            flag_name, 
            value, 
            description=description,
            org_id=org_id
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update feature flag")
        
        # Update adaptive router flags if applicable
        if flag_name in adaptive_router.features:
            adaptive_router.refresh_feature_flags(org_id)
        
        return {
            'success': True,
            'flag_name': flag_name,
            'new_value': value,
            'org_id': org_id,
            'timestamp': datetime.utcnow().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to set feature flag {flag_name}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update feature flag")


@router.post("/feature-flags/{flag_name}/toggle")
async def toggle_feature_flag(flag_name: str):
    """Toggle boolean feature flag."""
    try:
        success = feature_flag_manager.toggle_flag(flag_name)
        if not success:
            raise HTTPException(status_code=400, detail=f"Cannot toggle flag '{flag_name}' - not a boolean flag or doesn't exist")
        
        new_value = feature_flag_manager.get_flag(flag_name)
        
        # Update adaptive router flags if applicable
        if flag_name in adaptive_router.features:
            adaptive_router.refresh_feature_flags()
        
        return {
            'success': True,
            'flag_name': flag_name,
            'new_value': new_value,
            'timestamp': datetime.utcnow().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to toggle feature flag {flag_name}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to toggle feature flag")


@router.post("/feature-flags/{flag_name}/rollout")
async def update_rollout_percentage(
    flag_name: str,
    request: Dict[str, Any] = Body(...)
):
    """Update rollout percentage for gradual feature deployment."""
    try:
        percentage = request.get('percentage')
        if percentage is None or not (0 <= percentage <= 100):
            raise HTTPException(status_code=400, detail="Percentage must be between 0 and 100")
        
        success = feature_flag_manager.update_rollout_percentage(flag_name, percentage)
        if not success:
            raise HTTPException(status_code=404, detail=f"Feature flag '{flag_name}' not found")
        
        return {
            'success': True,
            'flag_name': flag_name,
            'new_rollout_percentage': percentage,
            'timestamp': datetime.utcnow().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update rollout for {flag_name}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update rollout percentage")


@router.get("/adaptive-router/metrics")
async def get_adaptive_router_metrics():
    """Get adaptive router performance metrics."""
    try:
        metrics = adaptive_router.get_metrics()
        return {
            'metrics': metrics,
            'timestamp': datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get router metrics: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve router metrics")


@router.get("/adaptive-router/config")
async def get_adaptive_router_config():
    """Get current adaptive router configuration."""
    try:
        return {
            'features': adaptive_router.features,
            'performance_targets': adaptive_router.performance_targets,
            'timestamp': datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get router config: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve router configuration")


@router.get("/dashboard")
async def get_monitoring_dashboard():
    """Get comprehensive monitoring dashboard data."""
    try:
        # Get performance stats
        perf_stats = performance_monitor.get_performance_stats(60)  # Last hour
        
        # Get feature flags
        feature_flags = feature_flag_manager.get_all_flags()
        
        # Get router metrics
        router_metrics = adaptive_router.get_metrics()
        
        # Get alerts
        alerts = performance_monitor.get_active_alerts()
        
        # Get health status
        health = performance_monitor.health_check()
        
        return {
            'dashboard': {
                'health': health,
                'performance': perf_stats,
                'feature_flags': {
                    'total_flags': len(feature_flags),
                    'enabled_flags': sum(1 for flag in feature_flags.values() if flag.get('value')),
                    'recent_flags': feature_flags  # Could be filtered for recent changes
                },
                'adaptive_router': router_metrics,
                'alerts': {
                    'total_alerts': len(alerts),
                    'critical_alerts': sum(1 for alert in alerts if alert.get('severity') == 'critical'),
                    'recent_alerts': alerts[-5:]  # Last 5 alerts
                }
            },
            'timestamp': datetime.utcnow().isoformat(),
            'refresh_interval_seconds': 30
        }
    except Exception as e:
        logger.error(f"Failed to get dashboard data: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve dashboard data")