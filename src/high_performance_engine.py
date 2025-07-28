import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import logging
from functools import partial
import time

from .traffic_profiles import TrafficModelEngine, TrafficProfile
from .performance_overlays import VectorizedOverlayEngine


def process_interval_chunk(chunk_args: Tuple) -> List[Dict[str, Any]]:
    """
    Process a chunk of intervals in a separate process.
    Optimized for maximum performance with minimal data transfer.
    """
    (start_idx, end_idx, profiles_config, config, overlay_data) = chunk_args
    
    # Create engines in the worker process (avoid pickling large objects)
    traffic_engine = TrafficModelEngine(config.get('random_seed', 42) + start_idx)
    profiles = traffic_engine.load_profiles(profiles_config)
    
    results = []
    
    for interval_idx in range(start_idx, end_idx):
        timestamp = interval_idx * config['interval_ms'] / 1000.0
        
        interval_stats = {
            'timestamp': timestamp,
            'interval_idx': interval_idx,
            'total_packets_ul': 0,
            'total_packets_dl': 0,
            'total_bytes_ul': 0,
            'total_bytes_dl': 0,
            'total_active_users': 0,
            'total_active_flows': 0,
            'profiles': []
        }
        
        # Process each profile with optimizations
        for profile in profiles:
            profile_stats = traffic_engine.generate_traffic_for_interval(
                profile,
                config['num_users'],
                config['bandwidth_per_user_mbps'],
                config['interval_ms'],
                interval_idx
            )
            
            # Apply overlays if provided (pre-computed data passed in)
            if overlay_data and profile.name in overlay_data:
                overlay_profile_data = overlay_data[profile.name]
                if interval_idx < len(overlay_profile_data['time_multipliers']):
                    # Apply time-of-day and event multipliers
                    time_mult = overlay_profile_data['time_multipliers'][interval_idx]
                    event_mult = overlay_profile_data['event_multipliers'][interval_idx]
                    combined_mult = time_mult * event_mult
                    
                    # Scale traffic
                    profile_stats['total_packets_ul'] = int(profile_stats['total_packets_ul'] * combined_mult)
                    profile_stats['total_packets_dl'] = int(profile_stats['total_packets_dl'] * combined_mult)
                    profile_stats['total_bytes_ul'] = int(profile_stats['total_bytes_ul'] * combined_mult)
                    profile_stats['total_bytes_dl'] = int(profile_stats['total_bytes_dl'] * combined_mult)
                    profile_stats['throughput_ul_mbps'] *= combined_mult
                    profile_stats['throughput_dl_mbps'] *= combined_mult
                    
                    # Add impairments
                    profile_stats['loss_rate_pct'] = overlay_profile_data['loss_patterns'][interval_idx] * 100
                    profile_stats['latency_ms'] = overlay_profile_data['latency_patterns'][interval_idx]
                    profile_stats['jitter_ms'] = overlay_profile_data['jitter_patterns'][interval_idx]
                    profile_stats['time_multiplier'] = time_mult
                    profile_stats['event_multiplier'] = event_mult
                    
                    # Apply packet loss
                    loss_factor = 1.0 - overlay_profile_data['loss_patterns'][interval_idx]
                    profile_stats['effective_packets_ul'] = int(profile_stats['total_packets_ul'] * loss_factor)
                    profile_stats['effective_packets_dl'] = int(profile_stats['total_packets_dl'] * loss_factor)
            
            # Aggregate totals
            interval_stats['total_packets_ul'] += profile_stats['total_packets_ul']
            interval_stats['total_packets_dl'] += profile_stats['total_packets_dl']
            interval_stats['total_bytes_ul'] += profile_stats['total_bytes_ul']
            interval_stats['total_bytes_dl'] += profile_stats['total_bytes_dl']
            interval_stats['total_active_users'] += profile_stats['active_users']
            interval_stats['total_active_flows'] += profile_stats['active_flows']
            interval_stats['profiles'].append(profile_stats)
        
        # Compute total throughput
        interval_stats['total_throughput_ul_mbps'] = sum(
            p.get('throughput_ul_mbps', 0) for p in interval_stats['profiles']
        )
        interval_stats['total_throughput_dl_mbps'] = sum(
            p.get('throughput_dl_mbps', 0) for p in interval_stats['profiles']
        )
        
        results.append(interval_stats)
    
    return results


class HighPerformanceSimulationEngine:
    """
    Ultra-high performance simulation engine with:
    - Aggressive multiprocessing 
    - Vectorized overlays
    - Memory-efficient data structures
    - Minimal Python loops
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Optimize core count for the workload
        self.num_cores = min(config.get('max_cores', 8), mp.cpu_count())
        
        # Initialize engines
        self.traffic_engine = TrafficModelEngine(config.get('random_seed', 42))
        self.profiles = self.traffic_engine.load_profiles(config['profiles'])
        
        # Initialize overlay engine if overlays are configured
        self.overlay_engine = None
        if self._has_overlays():
            self.overlay_engine = VectorizedOverlayEngine(config, config.get('random_seed', 42))
            self.logger.info("Initialized overlay engine with time-of-day and impairment patterns")
    
    def _has_overlays(self) -> bool:
        """Check if any overlay features are configured"""
        return bool(
            self.config.get('time_of_day') or 
            self.config.get('events') or 
            self.config.get('impairments')
        )
    
    def _prepare_overlay_data(self) -> Optional[Dict[str, Any]]:
        """Prepare overlay data for multiprocessing (serialize arrays)"""
        if not self.overlay_engine:
            return None
        
        overlay_data = {}
        for profile in self.profiles:
            profile_name = profile.name
            overlay_data[profile_name] = {
                'time_multipliers': self.overlay_engine.time_multipliers[profile_name],
                'event_multipliers': self.overlay_engine.event_multipliers[profile_name],
                'loss_patterns': self.overlay_engine.loss_patterns[profile_name],
                'latency_patterns': self.overlay_engine.latency_patterns[profile_name],
                'jitter_patterns': self.overlay_engine.jitter_patterns[profile_name]
            }
        
        return overlay_data
    
    def run_simulation(self) -> List[Dict[str, Any]]:
        """Run high-performance simulation with optimal multiprocessing"""
        start_time = time.time()
        
        total_intervals = int(self.config['duration_sec'] * 1000 / self.config['interval_ms'])
        self.logger.info(f"Starting simulation: {total_intervals} intervals, {self.num_cores} cores")
        
        # For small simulations, don't use multiprocessing (overhead)
        if total_intervals < 10 or self.config.get('disable_multiprocessing', False):
            return self._run_single_process(total_intervals)
        
        # Prepare data for multiprocessing
        profiles_config = self.config['profiles']
        overlay_data = self._prepare_overlay_data()
        
        # Optimal chunk sizing for performance
        min_chunk_size = max(1, total_intervals // (self.num_cores * 4))  # 4 chunks per core
        max_chunk_size = max(min_chunk_size, total_intervals // self.num_cores)
        chunk_size = min(max_chunk_size, 100)  # Cap at 100 intervals per chunk
        
        # Create chunks
        chunks = []
        for i in range(0, total_intervals, chunk_size):
            end_idx = min(i + chunk_size, total_intervals)
            chunks.append((i, end_idx, profiles_config, self.config, overlay_data))
        
        self.logger.info(f"Processing {len(chunks)} chunks with {chunk_size} intervals each (avg)")
        
        # Process chunks in parallel
        results = []
        with ProcessPoolExecutor(max_workers=self.num_cores) as executor:
            # Submit all chunks
            future_to_chunk = {
                executor.submit(process_interval_chunk, chunk): i 
                for i, chunk in enumerate(chunks)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_chunk):
                chunk_idx = future_to_chunk[future]
                try:
                    chunk_results = future.result()
                    results.extend(chunk_results)
                except Exception as e:
                    self.logger.error(f"Chunk {chunk_idx} failed: {e}")
                    raise
        
        # Sort by interval index to maintain order
        results.sort(key=lambda x: x['interval_idx'])
        
        elapsed = time.time() - start_time
        self.logger.info(f"Simulation completed in {elapsed:.2f}s ({total_intervals/elapsed:.0f} intervals/sec)")
        
        return results
    
    def _run_single_process(self, total_intervals: int) -> List[Dict[str, Any]]:
        """Single-process fallback for small simulations"""
        self.logger.info("Using single-process mode")
        
        results = []
        for interval_idx in range(total_intervals):
            timestamp = interval_idx * self.config['interval_ms'] / 1000.0
            
            interval_stats = {
                'timestamp': timestamp,
                'interval_idx': interval_idx,
                'total_packets_ul': 0,
                'total_packets_dl': 0,
                'total_bytes_ul': 0,
                'total_bytes_dl': 0,
                'total_active_users': 0,
                'total_active_flows': 0,
                'profiles': []
            }
            
            for profile in self.profiles:
                profile_stats = self.traffic_engine.generate_traffic_for_interval(
                    profile,
                    self.config['num_users'],
                    self.config['bandwidth_per_user_mbps'],
                    self.config['interval_ms'],
                    interval_idx
                )
                
                # Apply overlays if available
                if self.overlay_engine:
                    profile_stats = self.overlay_engine.apply_overlays(
                        profile.name, interval_idx, profile_stats
                    )
                
                # Aggregate
                interval_stats['total_packets_ul'] += profile_stats['total_packets_ul']
                interval_stats['total_packets_dl'] += profile_stats['total_packets_dl']
                interval_stats['total_bytes_ul'] += profile_stats['total_bytes_ul']
                interval_stats['total_bytes_dl'] += profile_stats['total_bytes_dl']
                interval_stats['total_active_users'] += profile_stats['active_users']
                interval_stats['total_active_flows'] += profile_stats['active_flows']
                interval_stats['profiles'].append(profile_stats)
            
            interval_stats['total_throughput_ul_mbps'] = sum(
                p.get('throughput_ul_mbps', 0) for p in interval_stats['profiles']
            )
            interval_stats['total_throughput_dl_mbps'] = sum(
                p.get('throughput_dl_mbps', 0) for p in interval_stats['profiles']
            )
            
            results.append(interval_stats)
        
        return results
    
    def run_vectorized_simulation(self) -> pd.DataFrame:
        """
        Ultra-fast vectorized simulation - processes entire time series at once.
        This is the fastest mode for large-scale simulations.
        """
        start_time = time.time()
        
        total_intervals = int(self.config['duration_sec'] * 1000 / self.config['interval_ms'])
        timestamps = np.arange(total_intervals) * self.config['interval_ms'] / 1000.0
        
        self.logger.info(f"Starting vectorized simulation: {total_intervals} intervals")
        
        # Pre-allocate result arrays for maximum performance
        num_profiles = len(self.profiles)
        
        # Initialize data arrays
        data_arrays = {
            'timestamp': timestamps,
            'interval_idx': np.arange(total_intervals),
            'total_packets_ul': np.zeros(total_intervals, dtype=np.int64),
            'total_packets_dl': np.zeros(total_intervals, dtype=np.int64),
            'total_bytes_ul': np.zeros(total_intervals, dtype=np.int64),
            'total_bytes_dl': np.zeros(total_intervals, dtype=np.int64),
            'total_active_users': np.zeros(total_intervals, dtype=np.int32),
            'total_active_flows': np.zeros(total_intervals, dtype=np.int32),
            'total_throughput_ul_mbps': np.zeros(total_intervals, dtype=np.float64),
            'total_throughput_dl_mbps': np.zeros(total_intervals, dtype=np.float64)
        }
        
        # Process each profile in vectorized manner
        for profile in self.profiles:
            self.logger.debug(f"Processing profile: {profile.name}")
            
            # Generate base traffic pattern
            interval_sec = self.config['interval_ms'] / 1000.0
            bytes_per_user = int(self.config['bandwidth_per_user_mbps'] * 1_000_000 * interval_sec / 8)
            base_bytes = int(bytes_per_user * self.config['num_users'] * profile.traffic_share)
            
            # Vectorized packet count calculation
            dist = profile.packet_distribution
            avg_packet_size = np.sum(dist.sizes * dist.probabilities)
            base_packets = max(1, int(base_bytes / avg_packet_size))
            
            # Apply burstiness pattern (vectorized)
            burst_phases = (np.arange(total_intervals) * 0.1) % (2 * np.pi)
            if profile.burstiness_factor > 1.0:
                burst_multipliers = 1.0 + (profile.burstiness_factor - 1.0) * (0.5 + 0.5 * np.sin(burst_phases))
                packet_counts = (base_packets * burst_multipliers).astype(np.int32)
            else:
                packet_counts = np.full(total_intervals, base_packets, dtype=np.int32)
            
            # Vectorized byte calculations
            byte_counts = packet_counts * avg_packet_size
            
            # UL/DL split (vectorized)
            ul_bytes = (byte_counts * profile.ul_dl_ratio / (1 + profile.ul_dl_ratio)).astype(np.int64)
            dl_bytes = byte_counts - ul_bytes
            ul_packets = (packet_counts * profile.ul_dl_ratio / (1 + profile.ul_dl_ratio)).astype(np.int32)
            dl_packets = packet_counts - ul_packets
            
            # Apply overlays if available (still vectorized)
            if self.overlay_engine:
                time_mult = self.overlay_engine.time_multipliers[profile.name]
                event_mult = self.overlay_engine.event_multipliers[profile.name]
                combined_mult = time_mult * event_mult
                
                ul_packets = (ul_packets * combined_mult).astype(np.int32)
                dl_packets = (dl_packets * combined_mult).astype(np.int32)
                ul_bytes = (ul_bytes * combined_mult).astype(np.int64)
                dl_bytes = (dl_bytes * combined_mult).astype(np.int64)
                
                # Store overlay data
                data_arrays[f'{profile.name}_time_multiplier'] = time_mult
                data_arrays[f'{profile.name}_event_multiplier'] = event_mult
                data_arrays[f'{profile.name}_loss_rate_pct'] = self.overlay_engine.loss_patterns[profile.name] * 100
                data_arrays[f'{profile.name}_latency_ms'] = self.overlay_engine.latency_patterns[profile.name]
                data_arrays[f'{profile.name}_jitter_ms'] = self.overlay_engine.jitter_patterns[profile.name]
            
            # Store profile-specific data
            data_arrays[f'{profile.name}_total_packets_ul'] = ul_packets
            data_arrays[f'{profile.name}_total_packets_dl'] = dl_packets
            data_arrays[f'{profile.name}_total_bytes_ul'] = ul_bytes
            data_arrays[f'{profile.name}_total_bytes_dl'] = dl_bytes
            data_arrays[f'{profile.name}_throughput_ul_mbps'] = (ul_bytes * 8) / (interval_sec * 1_000_000)
            data_arrays[f'{profile.name}_throughput_dl_mbps'] = (dl_bytes * 8) / (interval_sec * 1_000_000)
            data_arrays[f'{profile.name}_qos_5qi'] = np.full(total_intervals, profile.qos_5qi, dtype=np.int8)
            
            # Add to totals (ensure compatible types)
            data_arrays['total_packets_ul'] = data_arrays['total_packets_ul'] + ul_packets.astype(np.int64)
            data_arrays['total_packets_dl'] = data_arrays['total_packets_dl'] + dl_packets.astype(np.int64)
            data_arrays['total_bytes_ul'] = data_arrays['total_bytes_ul'] + ul_bytes.astype(np.int64)
            data_arrays['total_bytes_dl'] = data_arrays['total_bytes_dl'] + dl_bytes.astype(np.int64)
            data_arrays['total_throughput_ul_mbps'] = data_arrays['total_throughput_ul_mbps'] + (ul_bytes * 8) / (interval_sec * 1_000_000)
            data_arrays['total_throughput_dl_mbps'] = data_arrays['total_throughput_dl_mbps'] + (dl_bytes * 8) / (interval_sec * 1_000_000)
            
            # Users and flows
            active_users = min(self.config['num_users'], max(1, int(self.config['num_users'] * profile.traffic_share)))
            data_arrays['total_active_users'] = data_arrays['total_active_users'] + np.full(total_intervals, active_users, dtype=np.int32)
            data_arrays['total_active_flows'] = data_arrays['total_active_flows'] + np.full(total_intervals, active_users * profile.flows_per_user, dtype=np.int32)
        
        elapsed = time.time() - start_time
        self.logger.info(f"Vectorized simulation completed in {elapsed:.3f}s ({total_intervals/elapsed:.0f} intervals/sec)")
        
        return pd.DataFrame(data_arrays)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics and overlay summaries"""
        stats = {
            'num_cores': self.num_cores,
            'total_profiles': len(self.profiles),
            'has_overlays': self.overlay_engine is not None
        }
        
        if self.overlay_engine:
            stats['overlay_summary'] = self.overlay_engine.get_overlay_summary()
        
        return stats