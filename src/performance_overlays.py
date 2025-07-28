import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging


class VectorizedOverlayEngine:
    """
    High-performance vectorized overlay system for time-of-day variations,
    impairments, and event-driven patterns. All computations are vectorized
    for maximum performance on multi-core systems.
    """
    
    def __init__(self, config: Dict[str, Any], random_seed: int = 42):
        self.config = config
        self.rng = np.random.RandomState(random_seed)
        self.logger = logging.getLogger(__name__)
        
        # Pre-compute all overlay arrays
        self._precompute_overlays()
    
    def _precompute_overlays(self):
        """Pre-compute all time-varying overlays as NumPy arrays for vectorized operations"""
        total_intervals = int(self.config['duration_sec'] * 1000 / self.config['interval_ms'])
        
        # Time-of-day multipliers (vectorized for all intervals)
        self.time_multipliers = self._compute_time_of_day_multipliers(total_intervals)
        
        # Event multipliers (vectorized)
        self.event_multipliers = self._compute_event_multipliers(total_intervals)
        
        # Loss patterns (vectorized)
        self.loss_patterns = self._compute_loss_patterns(total_intervals)
        
        # Latency patterns (vectorized)
        self.latency_patterns = self._compute_latency_patterns(total_intervals)
        
        # Jitter patterns (vectorized)
        self.jitter_patterns = self._compute_jitter_patterns(total_intervals)
        
        self.logger.info(f"Pre-computed {total_intervals} intervals of overlay data")
    
    def _compute_time_of_day_multipliers(self, total_intervals: int) -> Dict[str, np.ndarray]:
        """Compute vectorized time-of-day multipliers for all profiles"""
        time_config = self.config.get('time_of_day', {})
        
        if not time_config:
            # Default flat multipliers
            return {profile['name']: np.ones(total_intervals) 
                   for profile in self.config['profiles']}
        
        # Create time array
        interval_sec = self.config['interval_ms'] / 1000.0
        start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        time_points = np.array([
            (start_time + timedelta(seconds=i * interval_sec)).hour + 
            (start_time + timedelta(seconds=i * interval_sec)).minute / 60.0
            for i in range(total_intervals)
        ])
        
        multipliers = {}
        for profile in self.config['profiles']:
            profile_name = profile['name']
            
            if profile_name in time_config:
                # Use configured hourly pattern
                hourly_pattern = np.array(time_config[profile_name])
                # Interpolate to get multipliers for each interval
                hour_indices = (time_points % 24).astype(int)
                multipliers[profile_name] = hourly_pattern[hour_indices]
            else:
                # Default business hours pattern
                multipliers[profile_name] = self._default_business_pattern(time_points)
        
        return multipliers
    
    def _default_business_pattern(self, time_points: np.ndarray) -> np.ndarray:
        """Generate default business hours pattern"""
        # Peak hours: 9-12, 14-18, Evening: 19-22
        pattern = np.ones_like(time_points) * 0.3  # Night baseline
        
        # Morning ramp
        morning_mask = (time_points >= 6) & (time_points < 9)
        pattern[morning_mask] = 0.3 + 0.7 * (time_points[morning_mask] - 6) / 3
        
        # Business hours peaks
        business_mask = ((time_points >= 9) & (time_points < 12)) | ((time_points >= 14) & (time_points < 18))
        pattern[business_mask] = 1.0
        
        # Lunch dip
        lunch_mask = (time_points >= 12) & (time_points < 14)
        pattern[lunch_mask] = 0.7
        
        # Evening peak
        evening_mask = (time_points >= 19) & (time_points < 22)
        pattern[evening_mask] = 1.2
        
        return pattern
    
    def _compute_event_multipliers(self, total_intervals: int) -> Dict[str, np.ndarray]:
        """Compute vectorized event multipliers"""
        event_config = self.config.get('events', {})
        
        multipliers = {}
        for profile in self.config['profiles']:
            profile_name = profile['name']
            base_multiplier = np.ones(total_intervals)
            
            if profile_name in event_config:
                # Apply event spikes at specified intervals
                for event in event_config[profile_name]:
                    start_idx = int(event.get('start_interval', 0))
                    end_idx = int(event.get('end_interval', start_idx + 1))
                    multiplier = event.get('multiplier', 1.0)
                    
                    start_idx = max(0, start_idx)
                    end_idx = min(total_intervals, end_idx)
                    
                    base_multiplier[start_idx:end_idx] *= multiplier
            
            multipliers[profile_name] = base_multiplier
        
        return multipliers
    
    def _compute_loss_patterns(self, total_intervals: int) -> Dict[str, np.ndarray]:
        """Compute vectorized packet loss patterns"""
        impairments = self.config.get('impairments', {})
        
        patterns = {}
        for profile in self.config['profiles']:
            profile_name = profile['name']
            
            if profile_name in impairments and 'loss_rate' in impairments[profile_name]:
                base_loss = impairments[profile_name]['loss_rate']
                # Add small random variation
                variation = self.rng.normal(0, base_loss * 0.1, total_intervals)
                patterns[profile_name] = np.clip(base_loss + variation, 0, 1.0)
            else:
                patterns[profile_name] = np.zeros(total_intervals)
        
        return patterns
    
    def _compute_latency_patterns(self, total_intervals: int) -> Dict[str, np.ndarray]:
        """Compute vectorized latency patterns"""
        impairments = self.config.get('impairments', {})
        
        patterns = {}
        for profile in self.config['profiles']:
            profile_name = profile['name']
            
            if profile_name in impairments and 'latency_ms' in impairments[profile_name]:
                latency_cfg = impairments[profile_name]['latency_ms']
                mean_latency = latency_cfg.get('mean', 10)
                std_latency = latency_cfg.get('std', 1)
                
                # Generate correlated latency pattern (avoid completely random)
                base_pattern = self.rng.normal(mean_latency, std_latency, total_intervals)
                # Apply smoothing for realistic correlation
                kernel = np.array([0.2, 0.6, 0.2])
                if total_intervals > 2:
                    smoothed = np.convolve(base_pattern, kernel, mode='same')
                    patterns[profile_name] = np.clip(smoothed, 0, None)
                else:
                    patterns[profile_name] = np.clip(base_pattern, 0, None)
            else:
                # Default latency based on profile type
                default_latencies = {'eMBB': 15, 'URLLC': 1, 'mMTC': 100, 'VoNR': 20}
                default_latency = default_latencies.get(profile_name, 10)
                patterns[profile_name] = np.full(total_intervals, default_latency, dtype=float)
        
        return patterns
    
    def _compute_jitter_patterns(self, total_intervals: int) -> Dict[str, np.ndarray]:
        """Compute vectorized jitter patterns"""
        impairments = self.config.get('impairments', {})
        
        patterns = {}
        for profile in self.config['profiles']:
            profile_name = profile['name']
            
            if profile_name in impairments and 'jitter_ms' in impairments[profile_name]:
                jitter_cfg = impairments[profile_name]['jitter_ms']
                std_jitter = jitter_cfg.get('std', 0.5)
                
                # Jitter is typically the variation in latency
                jitter_values = np.abs(self.rng.normal(0, std_jitter, total_intervals))
                patterns[profile_name] = jitter_values
            else:
                # Default jitter based on profile type
                default_jitters = {'eMBB': 2.0, 'URLLC': 0.1, 'mMTC': 10.0, 'VoNR': 1.0}
                default_jitter = default_jitters.get(profile_name, 1.0)
                patterns[profile_name] = np.full(total_intervals, default_jitter, dtype=float)
        
        return patterns
    
    def apply_overlays(self, profile_name: str, interval_idx: int, base_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Apply all overlays to base statistics - optimized for vectorized access"""
        
        # Time-of-day multiplier
        time_mult = self.time_multipliers[profile_name][interval_idx]
        
        # Event multiplier  
        event_mult = self.event_multipliers[profile_name][interval_idx]
        
        # Combined traffic multiplier
        traffic_mult = time_mult * event_mult
        
        # Apply traffic scaling
        enhanced_stats = base_stats.copy()
        enhanced_stats['total_packets_ul'] = int(base_stats['total_packets_ul'] * traffic_mult)
        enhanced_stats['total_packets_dl'] = int(base_stats['total_packets_dl'] * traffic_mult)
        enhanced_stats['total_bytes_ul'] = int(base_stats['total_bytes_ul'] * traffic_mult)
        enhanced_stats['total_bytes_dl'] = int(base_stats['total_bytes_dl'] * traffic_mult)
        enhanced_stats['throughput_ul_mbps'] = base_stats['throughput_ul_mbps'] * traffic_mult
        enhanced_stats['throughput_dl_mbps'] = base_stats['throughput_dl_mbps'] * traffic_mult
        
        # Apply impairments
        enhanced_stats['loss_rate_pct'] = self.loss_patterns[profile_name][interval_idx] * 100
        enhanced_stats['latency_ms'] = self.latency_patterns[profile_name][interval_idx]
        enhanced_stats['jitter_ms'] = self.jitter_patterns[profile_name][interval_idx]
        
        # Apply loss to packet counts
        loss_factor = 1.0 - self.loss_patterns[profile_name][interval_idx]
        enhanced_stats['effective_packets_ul'] = int(enhanced_stats['total_packets_ul'] * loss_factor)
        enhanced_stats['effective_packets_dl'] = int(enhanced_stats['total_packets_dl'] * loss_factor)
        
        # Store multipliers for analysis
        enhanced_stats['time_multiplier'] = time_mult
        enhanced_stats['event_multiplier'] = event_mult
        enhanced_stats['combined_multiplier'] = traffic_mult
        
        return enhanced_stats
    
    def get_overlay_summary(self) -> Dict[str, Any]:
        """Get summary statistics of all overlays for reporting"""
        summary = {
            'time_of_day_stats': {},
            'event_stats': {},
            'impairment_stats': {}
        }
        
        for profile_name in [p['name'] for p in self.config['profiles']]:
            # Time-of-day stats
            time_mult = self.time_multipliers[profile_name]
            summary['time_of_day_stats'][profile_name] = {
                'min_multiplier': float(np.min(time_mult)),
                'max_multiplier': float(np.max(time_mult)),
                'avg_multiplier': float(np.mean(time_mult)),
                'peak_hours_boost': float(np.max(time_mult) / np.mean(time_mult))
            }
            
            # Event stats
            event_mult = self.event_multipliers[profile_name]
            summary['event_stats'][profile_name] = {
                'has_events': bool(np.any(event_mult > 1.0)),
                'max_event_multiplier': float(np.max(event_mult)),
                'event_intervals': int(np.sum(event_mult > 1.0))
            }
            
            # Impairment stats
            summary['impairment_stats'][profile_name] = {
                'avg_loss_pct': float(np.mean(self.loss_patterns[profile_name]) * 100),
                'avg_latency_ms': float(np.mean(self.latency_patterns[profile_name])),
                'avg_jitter_ms': float(np.mean(self.jitter_patterns[profile_name]))
            }
        
        return summary