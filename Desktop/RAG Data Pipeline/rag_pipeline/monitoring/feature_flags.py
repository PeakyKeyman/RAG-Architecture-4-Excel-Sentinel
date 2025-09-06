"""Feature flag management system for adaptive RAG pipeline."""

import json
import os
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum

from ..core.safe_logging import get_logger


class FeatureFlagType(Enum):
    """Types of feature flags."""
    BOOLEAN = "boolean"
    STRING = "string" 
    INTEGER = "integer"
    FLOAT = "float"
    PERCENTAGE = "percentage"


@dataclass
class FeatureFlag:
    """Individual feature flag configuration."""
    name: str
    value: Any
    flag_type: FeatureFlagType
    description: str
    enabled: bool = True
    created_at: datetime = None
    updated_at: datetime = None
    rollout_percentage: float = 100.0
    user_groups: List[str] = None
    org_overrides: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()
        if self.user_groups is None:
            self.user_groups = []
        if self.org_overrides is None:
            self.org_overrides = {}


class FeatureFlagManager:
    """Centralized feature flag management system."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.logger = get_logger(__name__)
        
        # Thread-safe storage
        self._lock = threading.RLock()
        self._flags: Dict[str, FeatureFlag] = {}
        self._callbacks: Dict[str, List[Callable]] = {}
        
        # Load configuration
        self._load_flags()
        
        # Initialize default flags
        self._initialize_default_flags()
        
        self.logger.info("Feature flag manager initialized", extra={
            'total_flags': len(self._flags),
            'config_path': config_path
        })
    
    def _load_flags(self):
        """Load feature flags from configuration file."""
        if not self.config_path or not os.path.exists(self.config_path):
            return
        
        try:
            with open(self.config_path, 'r') as f:
                data = json.load(f)
            
            for flag_data in data.get('feature_flags', []):
                flag = FeatureFlag(
                    name=flag_data['name'],
                    value=flag_data['value'],
                    flag_type=FeatureFlagType(flag_data.get('type', 'boolean')),
                    description=flag_data.get('description', ''),
                    enabled=flag_data.get('enabled', True),
                    rollout_percentage=flag_data.get('rollout_percentage', 100.0),
                    user_groups=flag_data.get('user_groups', []),
                    org_overrides=flag_data.get('org_overrides', {})
                )
                self._flags[flag.name] = flag
                
        except Exception as e:
            self.logger.error("Failed to load feature flags", extra={
                'error': str(e),
                'config_path': self.config_path
            })
    
    def _initialize_default_flags(self):
        """Initialize default feature flags for the system."""
        default_flags = [
            # Adaptive RAG flags
            FeatureFlag(
                name='adaptive_routing_enabled',
                value=True,
                flag_type=FeatureFlagType.BOOLEAN,
                description='Enable adaptive query routing based on complexity'
            ),
            FeatureFlag(
                name='simple_query_optimization',
                value=True,
                flag_type=FeatureFlagType.BOOLEAN,
                description='Use direct vector search for simple queries'
            ),
            FeatureFlag(
                name='complex_query_ensemble', 
                value=True,
                flag_type=FeatureFlagType.BOOLEAN,
                description='Use full ensemble strategy for complex queries'
            ),
            FeatureFlag(
                name='performance_monitoring',
                value=True,
                flag_type=FeatureFlagType.BOOLEAN,
                description='Enable detailed performance monitoring'
            ),
            
            # Temporal understanding flags
            FeatureFlag(
                name='temporal_scoring_enabled',
                value=True,
                flag_type=FeatureFlagType.BOOLEAN,
                description='Enable temporal relevance scoring'
            ),
            FeatureFlag(
                name='temporal_query_analysis',
                value=True,
                flag_type=FeatureFlagType.BOOLEAN,
                description='Analyze queries for temporal intent'
            ),
            FeatureFlag(
                name='document_date_extraction',
                value=True,
                flag_type=FeatureFlagType.BOOLEAN,
                description='Extract dates from document content'
            ),
            
            # Performance tuning flags
            FeatureFlag(
                name='classification_caching',
                value=True,
                flag_type=FeatureFlagType.BOOLEAN,
                description='Cache query classifications'
            ),
            FeatureFlag(
                name='result_deduplication',
                value=True,
                flag_type=FeatureFlagType.BOOLEAN,
                description='Remove duplicate results from searches'
            ),
            
            # A/B testing flags
            FeatureFlag(
                name='new_reranking_algorithm',
                value=False,
                flag_type=FeatureFlagType.BOOLEAN,
                description='Test new reranking algorithm',
                rollout_percentage=10.0
            ),
            FeatureFlag(
                name='enhanced_temporal_scoring',
                value=False,
                flag_type=FeatureFlagType.BOOLEAN,
                description='Enhanced temporal scoring with business context',
                rollout_percentage=25.0
            )
        ]
        
        for flag in default_flags:
            if flag.name not in self._flags:
                self._flags[flag.name] = flag
    
    def get_flag(self, name: str, default: Any = None, org_id: str = None, 
                 user_id: str = None) -> Any:
        """Get feature flag value with org/user overrides."""
        with self._lock:
            flag = self._flags.get(name)
            
            if not flag:
                return default
            
            if not flag.enabled:
                return default if flag.flag_type == FeatureFlagType.BOOLEAN else flag.value
            
            # Check org-specific overrides
            if org_id and org_id in flag.org_overrides:
                return flag.org_overrides[org_id]
            
            # Check rollout percentage for gradual rollout
            if flag.rollout_percentage < 100.0:
                # Use consistent hashing for user/org to ensure stable rollout
                hash_input = user_id or org_id or "default"
                hash_val = hash(f"{name}:{hash_input}") % 100
                if hash_val >= flag.rollout_percentage:
                    return default if flag.flag_type == FeatureFlagType.BOOLEAN else None
            
            return flag.value
    
    def set_flag(self, name: str, value: Any, description: str = None,
                 org_id: str = None) -> bool:
        """Set or update a feature flag value."""
        try:
            with self._lock:
                if name in self._flags:
                    flag = self._flags[name]
                    
                    # Handle org-specific overrides
                    if org_id:
                        flag.org_overrides[org_id] = value
                    else:
                        old_value = flag.value
                        flag.value = value
                        
                        # Update description if provided
                        if description:
                            flag.description = description
                        
                        flag.updated_at = datetime.utcnow()
                        
                        # Notify callbacks
                        self._notify_callbacks(name, old_value, value)
                else:
                    # Create new flag
                    flag_type = self._infer_flag_type(value)
                    flag = FeatureFlag(
                        name=name,
                        value=value,
                        flag_type=flag_type,
                        description=description or f"Auto-created flag: {name}"
                    )
                    self._flags[name] = flag
            
            self._save_flags()
            return True
            
        except Exception as e:
            self.logger.error("Failed to set feature flag", extra={
                'flag_name': name,
                'error': str(e)
            })
            return False
    
    def toggle_flag(self, name: str) -> bool:
        """Toggle a boolean feature flag."""
        flag = self._flags.get(name)
        if not flag or flag.flag_type != FeatureFlagType.BOOLEAN:
            return False
        
        return self.set_flag(name, not flag.value)
    
    def get_all_flags(self, org_id: str = None) -> Dict[str, Any]:
        """Get all feature flags with their current values."""
        with self._lock:
            result = {}
            for name, flag in self._flags.items():
                result[name] = {
                    'value': self.get_flag(name, org_id=org_id),
                    'type': flag.flag_type.value,
                    'description': flag.description,
                    'enabled': flag.enabled,
                    'rollout_percentage': flag.rollout_percentage,
                    'updated_at': flag.updated_at.isoformat() if flag.updated_at else None
                }
            return result
    
    def update_rollout_percentage(self, name: str, percentage: float) -> bool:
        """Update rollout percentage for gradual feature deployment."""
        if not 0 <= percentage <= 100:
            return False
        
        with self._lock:
            flag = self._flags.get(name)
            if not flag:
                return False
            
            old_percentage = flag.rollout_percentage
            flag.rollout_percentage = percentage
            flag.updated_at = datetime.utcnow()
            
            self.logger.info("Updated rollout percentage", extra={
                'flag_name': name,
                'old_percentage': old_percentage,
                'new_percentage': percentage
            })
        
        self._save_flags()
        return True
    
    def enable_flag(self, name: str) -> bool:
        """Enable a feature flag."""
        with self._lock:
            flag = self._flags.get(name)
            if not flag:
                return False
            
            flag.enabled = True
            flag.updated_at = datetime.utcnow()
        
        self._save_flags()
        return True
    
    def disable_flag(self, name: str) -> bool:
        """Disable a feature flag."""
        with self._lock:
            flag = self._flags.get(name)
            if not flag:
                return False
            
            flag.enabled = False
            flag.updated_at = datetime.utcnow()
        
        self._save_flags()
        return True
    
    def register_callback(self, flag_name: str, callback: Callable[[str, Any, Any], None]):
        """Register callback for flag value changes."""
        with self._lock:
            if flag_name not in self._callbacks:
                self._callbacks[flag_name] = []
            self._callbacks[flag_name].append(callback)
    
    def _notify_callbacks(self, flag_name: str, old_value: Any, new_value: Any):
        """Notify registered callbacks of flag changes."""
        callbacks = self._callbacks.get(flag_name, [])
        for callback in callbacks:
            try:
                callback(flag_name, old_value, new_value)
            except Exception as e:
                self.logger.error("Feature flag callback failed", extra={
                    'flag_name': flag_name,
                    'error': str(e)
                })
    
    def _infer_flag_type(self, value: Any) -> FeatureFlagType:
        """Infer feature flag type from value."""
        if isinstance(value, bool):
            return FeatureFlagType.BOOLEAN
        elif isinstance(value, int):
            return FeatureFlagType.INTEGER
        elif isinstance(value, float):
            return FeatureFlagType.FLOAT
        else:
            return FeatureFlagType.STRING
    
    def _save_flags(self):
        """Save feature flags to configuration file."""
        if not self.config_path:
            return
        
        try:
            data = {
                'feature_flags': [
                    {
                        'name': flag.name,
                        'value': flag.value,
                        'type': flag.flag_type.value,
                        'description': flag.description,
                        'enabled': flag.enabled,
                        'rollout_percentage': flag.rollout_percentage,
                        'user_groups': flag.user_groups,
                        'org_overrides': flag.org_overrides,
                        'created_at': flag.created_at.isoformat() if flag.created_at else None,
                        'updated_at': flag.updated_at.isoformat() if flag.updated_at else None
                    }
                    for flag in self._flags.values()
                ]
            }
            
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.error("Failed to save feature flags", extra={
                'error': str(e),
                'config_path': self.config_path
            })
    
    def export_config(self) -> Dict[str, Any]:
        """Export current feature flag configuration."""
        with self._lock:
            return {
                'total_flags': len(self._flags),
                'flags': {
                    name: {
                        'value': flag.value,
                        'type': flag.flag_type.value,
                        'enabled': flag.enabled,
                        'rollout_percentage': flag.rollout_percentage,
                        'description': flag.description
                    }
                    for name, flag in self._flags.items()
                },
                'exported_at': datetime.utcnow().isoformat()
            }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on feature flag system."""
        with self._lock:
            total_flags = len(self._flags)
            enabled_flags = sum(1 for flag in self._flags.values() if flag.enabled)
            partial_rollout = sum(1 for flag in self._flags.values() 
                                if flag.rollout_percentage < 100.0)
        
        return {
            'status': 'healthy',
            'total_flags': total_flags,
            'enabled_flags': enabled_flags,
            'partial_rollout_flags': partial_rollout,
            'config_loaded': self.config_path is not None,
            'timestamp': datetime.utcnow().isoformat()
        }


# Global feature flag manager instance
_flag_manager_instance: Optional[FeatureFlagManager] = None
_flag_manager_lock = threading.Lock()


def get_feature_flag_manager(config_path: str = None) -> FeatureFlagManager:
    """Get or create the global feature flag manager instance."""
    global _flag_manager_instance
    
    with _flag_manager_lock:
        if _flag_manager_instance is None:
            _flag_manager_instance = FeatureFlagManager(config_path)
        return _flag_manager_instance