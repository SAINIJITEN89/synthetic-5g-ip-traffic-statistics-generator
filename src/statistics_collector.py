import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class IntervalStatistics:
    timestamp: float
    interval_idx: int
    total_packets_ul: int
    total_packets_dl: int
    total_bytes_ul: int
    total_bytes_dl: int
    total_active_users: int
    total_active_flows: int
    total_throughput_ul_mbps: float
    total_throughput_dl_mbps: float
    profile_stats: Dict[str, Any]


class StatisticsCollector:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.interval_ms = config['interval_ms']
        self.collected_stats: List[IntervalStatistics] = []
    
    def collect_interval_stats(self, interval_data: Dict[str, Any]) -> IntervalStatistics:
        profile_breakdown = {}
        
        if 'profiles' in interval_data:
            for profile_stat in interval_data['profiles']:
                profile_name = profile_stat['profile_name']
                profile_breakdown[profile_name] = profile_stat
        
        stats = IntervalStatistics(
            timestamp=interval_data['timestamp'],
            interval_idx=interval_data['interval_idx'],
            total_packets_ul=interval_data['total_packets_ul'],
            total_packets_dl=interval_data['total_packets_dl'],
            total_bytes_ul=interval_data['total_bytes_ul'],
            total_bytes_dl=interval_data['total_bytes_dl'],
            total_active_users=interval_data['total_active_users'],
            total_active_flows=interval_data['total_active_flows'],
            total_throughput_ul_mbps=interval_data.get('total_throughput_ul_mbps', 0),
            total_throughput_dl_mbps=interval_data.get('total_throughput_dl_mbps', 0),
            profile_stats=profile_breakdown
        )
        
        self.collected_stats.append(stats)
        return stats
    
    def calculate_summary_statistics(self) -> Dict[str, Any]:
        if not self.collected_stats:
            return {}
        
        df = self.to_dataframe()
        
        summary = {
            'simulation': {
                'total_duration_sec': len(self.collected_stats) * self.interval_ms / 1000.0,
                'total_intervals': len(self.collected_stats),
                'interval_ms': self.interval_ms
            },
            'traffic_totals': {
                'total_packets_ul': int(df['total_packets_ul'].sum()),
                'total_packets_dl': int(df['total_packets_dl'].sum()),
                'total_bytes_ul': int(df['total_bytes_ul'].sum()),
                'total_bytes_dl': int(df['total_bytes_dl'].sum()),
                'avg_throughput_ul_mbps': float(df['total_throughput_ul_mbps'].mean()),
                'avg_throughput_dl_mbps': float(df['total_throughput_dl_mbps'].mean()),
                'peak_throughput_ul_mbps': float(df['total_throughput_ul_mbps'].max()),
                'peak_throughput_dl_mbps': float(df['total_throughput_dl_mbps'].max())
            },
            'user_flow_stats': {
                'avg_active_users': float(df['total_active_users'].mean()),
                'max_active_users': int(df['total_active_users'].max()),
                'avg_active_flows': float(df['total_active_flows'].mean()),
                'max_active_flows': int(df['total_active_flows'].max())
            }
        }
        
        profile_summaries = {}
        for stat in self.collected_stats:
            for profile_name, profile_data in stat.profile_stats.items():
                if profile_name not in profile_summaries:
                    profile_summaries[profile_name] = {
                        'packets_ul': [],
                        'packets_dl': [],
                        'bytes_ul': [],
                        'bytes_dl': [],
                        'throughput_ul': [],
                        'throughput_dl': [],
                        'active_users': [],
                        'qos_5qi': profile_data.get('qos_5qi', 0)
                    }
                
                profile_summaries[profile_name]['packets_ul'].append(profile_data.get('total_packets_ul', 0))
                profile_summaries[profile_name]['packets_dl'].append(profile_data.get('total_packets_dl', 0))
                profile_summaries[profile_name]['bytes_ul'].append(profile_data.get('total_bytes_ul', 0))
                profile_summaries[profile_name]['bytes_dl'].append(profile_data.get('total_bytes_dl', 0))
                profile_summaries[profile_name]['throughput_ul'].append(profile_data.get('throughput_ul_mbps', 0))
                profile_summaries[profile_name]['throughput_dl'].append(profile_data.get('throughput_dl_mbps', 0))
                profile_summaries[profile_name]['active_users'].append(profile_data.get('active_users', 0))
        
        for profile_name, data in profile_summaries.items():
            summary[f'profile_{profile_name}'] = {
                'total_packets_ul': sum(data['packets_ul']),
                'total_packets_dl': sum(data['packets_dl']),
                'total_bytes_ul': sum(data['bytes_ul']),
                'total_bytes_dl': sum(data['bytes_dl']),
                'avg_throughput_ul_mbps': np.mean(data['throughput_ul']),
                'avg_throughput_dl_mbps': np.mean(data['throughput_dl']),
                'avg_active_users': np.mean(data['active_users']),
                'qos_5qi': data['qos_5qi']
            }
        
        return summary
    
    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        
        for stat in self.collected_stats:
            row = {
                'timestamp': stat.timestamp,
                'interval_idx': stat.interval_idx,
                'total_packets_ul': stat.total_packets_ul,
                'total_packets_dl': stat.total_packets_dl,
                'total_bytes_ul': stat.total_bytes_ul,
                'total_bytes_dl': stat.total_bytes_dl,
                'total_active_users': stat.total_active_users,
                'total_active_flows': stat.total_active_flows,
                'total_throughput_ul_mbps': stat.total_throughput_ul_mbps,
                'total_throughput_dl_mbps': stat.total_throughput_dl_mbps
            }
            
            for profile_name, profile_data in stat.profile_stats.items():
                for key, value in profile_data.items():
                    if key != 'profile_name':
                        column_name = f"{profile_name}_{key}"
                        row[column_name] = value
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def get_histogram_data(self, metric: str, bins: int = 20) -> Dict[str, Any]:
        if not self.collected_stats:
            return {}
        
        df = self.to_dataframe()
        if metric not in df.columns:
            return {}
        
        hist, bin_edges = np.histogram(df[metric], bins=bins)
        
        return {
            'metric': metric,
            'bins': bins,
            'counts': hist.tolist(),
            'bin_edges': bin_edges.tolist(),
            'min_value': float(df[metric].min()),
            'max_value': float(df[metric].max()),
            'mean_value': float(df[metric].mean()),
            'std_value': float(df[metric].std())
        }
    
    def clear_stats(self):
        self.collected_stats.clear()