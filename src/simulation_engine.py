import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from .traffic_profiles import TrafficModelEngine, TrafficProfile


class SimulationEngine:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.traffic_engine = TrafficModelEngine(config.get('random_seed', 42))
        self.profiles = self.traffic_engine.load_profiles(config['profiles'])
        self.num_cores = min(8, mp.cpu_count())
    
    def _simulate_chunk(self, chunk_args: tuple) -> List[Dict[str, Any]]:
        start_interval, end_interval, profiles, config = chunk_args
        
        traffic_engine = TrafficModelEngine(config.get('random_seed', 42))
        results = []
        
        for interval_idx in range(start_interval, end_interval):
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
            
            for profile in profiles:
                profile_stats = traffic_engine.generate_traffic_for_interval(
                    profile,
                    config['num_users'],
                    config['bandwidth_per_user_mbps'],
                    config['interval_ms']
                )
                
                interval_stats['total_packets_ul'] += profile_stats['total_packets_ul']
                interval_stats['total_packets_dl'] += profile_stats['total_packets_dl']
                interval_stats['total_bytes_ul'] += profile_stats['total_bytes_ul']
                interval_stats['total_bytes_dl'] += profile_stats['total_bytes_dl']
                interval_stats['total_active_users'] += profile_stats['active_users']
                interval_stats['total_active_flows'] += profile_stats['active_flows']
                interval_stats['profiles'].append(profile_stats)
            
            interval_stats['total_throughput_ul_mbps'] = sum(
                p['throughput_ul_mbps'] for p in interval_stats['profiles']
            )
            interval_stats['total_throughput_dl_mbps'] = sum(
                p['throughput_dl_mbps'] for p in interval_stats['profiles']
            )
            
            results.append(interval_stats)
        
        return results
    
    def run_simulation(self) -> List[Dict[str, Any]]:
        total_intervals = int(self.config['duration_sec'] * 1000 / self.config['interval_ms'])
        
        if total_intervals <= self.num_cores or self.config.get('disable_multiprocessing', False):
            return self._simulate_chunk((0, total_intervals, self.profiles, self.config))
        
        chunk_size = max(1, total_intervals // self.num_cores)
        chunks = []
        
        for i in range(0, total_intervals, chunk_size):
            end_idx = min(i + chunk_size, total_intervals)
            chunks.append((i, end_idx, self.profiles, self.config))
        
        with ProcessPoolExecutor(max_workers=self.num_cores) as executor:
            chunk_results = list(executor.map(self._simulate_chunk, chunks))
        
        all_results = []
        for chunk_result in chunk_results:
            all_results.extend(chunk_result)
        
        return sorted(all_results, key=lambda x: x['interval_idx'])
    
    def run_vectorized_simulation(self) -> pd.DataFrame:
        total_intervals = int(self.config['duration_sec'] * 1000 / self.config['interval_ms'])
        timestamps = np.arange(total_intervals) * self.config['interval_ms'] / 1000.0
        
        all_data = []
        
        for interval_idx in range(total_intervals):
            row_data = {
                'timestamp': timestamps[interval_idx],
                'interval_idx': interval_idx,
                'total_packets_ul': 0,
                'total_packets_dl': 0,
                'total_bytes_ul': 0,
                'total_bytes_dl': 0,
                'total_active_users': 0,
                'total_active_flows': 0,
                'total_throughput_ul_mbps': 0,
                'total_throughput_dl_mbps': 0
            }
            
            for profile in self.profiles:
                profile_stats = self.traffic_engine.generate_traffic_for_interval(
                    profile,
                    self.config['num_users'],
                    self.config['bandwidth_per_user_mbps'],
                    self.config['interval_ms']
                )
                
                row_data['total_packets_ul'] += profile_stats['total_packets_ul']
                row_data['total_packets_dl'] += profile_stats['total_packets_dl']
                row_data['total_bytes_ul'] += profile_stats['total_bytes_ul']
                row_data['total_bytes_dl'] += profile_stats['total_bytes_dl']
                row_data['total_active_users'] += profile_stats['active_users']
                row_data['total_active_flows'] += profile_stats['active_flows']
                row_data['total_throughput_ul_mbps'] += profile_stats['throughput_ul_mbps']
                row_data['total_throughput_dl_mbps'] += profile_stats['throughput_dl_mbps']
                
                for key, value in profile_stats.items():
                    if key not in ['active_users', 'active_flows']:
                        profile_key = f"{profile.name}_{key}"
                        row_data[profile_key] = value
            
            all_data.append(row_data)
        
        return pd.DataFrame(all_data)