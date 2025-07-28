import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field


@dataclass
class PacketSizeDistribution:
    """Optimized packet size distribution for vectorized sampling"""
    sizes: np.ndarray
    probabilities: np.ndarray
    cumulative_probs: np.ndarray = field(init=False)
    
    def __post_init__(self):
        self.cumulative_probs = np.cumsum(self.probabilities)
    
    @classmethod
    def from_config(cls, packet_size_bytes: List[int], packet_size_pct: List[float]):
        sizes = np.array(packet_size_bytes, dtype=int)
        probs = np.array(packet_size_pct, dtype=float) / 100.0
        return cls(sizes=sizes, probabilities=probs)


@dataclass
class TrafficProfile:
    name: str
    traffic_share: float
    packet_size_bytes: List[int]
    packet_size_pct: List[float]
    ul_dl_ratio: float
    flows_per_user: int
    qos_5qi: int
    
    # Enhanced attributes for traffic mix reflections
    protocol: str = "TCP"
    priority: int = 1
    burstiness_factor: float = 1.0
    session_duration_sec: Optional[float] = None
    inter_arrival_pattern: str = "poisson"
    
    # Cached distributions for performance
    _packet_dist: Optional[PacketSizeDistribution] = field(default=None, init=False)
    
    def __post_init__(self):
        if len(self.packet_size_bytes) != len(self.packet_size_pct):
            raise ValueError("Packet size and percentage lists must have same length")
        if abs(sum(self.packet_size_pct) - 100) > 0.01:
            raise ValueError("Packet size percentages must sum to 100")
        
        # Pre-compute packet size distribution for performance
        self._packet_dist = PacketSizeDistribution.from_config(
            self.packet_size_bytes, self.packet_size_pct
        )
    
    @property
    def packet_distribution(self) -> PacketSizeDistribution:
        return self._packet_dist


class TrafficModelEngine:
    def __init__(self, random_seed: int = 42):
        self.rng = np.random.RandomState(random_seed)
        self.builtin_profiles = self._create_builtin_profiles()
    
    def _create_builtin_profiles(self) -> Dict[str, TrafficProfile]:
        return {
            'eMBB': TrafficProfile(
                name='eMBB',
                traffic_share=0.85,
                packet_size_bytes=[1400, 200],
                packet_size_pct=[88, 12],
                ul_dl_ratio=0.12,
                flows_per_user=2,
                qos_5qi=9
            ),
            'URLLC': TrafficProfile(
                name='URLLC',
                traffic_share=0.10,
                packet_size_bytes=[88],
                packet_size_pct=[100],
                ul_dl_ratio=1.0,
                flows_per_user=1,
                qos_5qi=3
            ),
            'mMTC': TrafficProfile(
                name='mMTC',
                traffic_share=0.05,
                packet_size_bytes=[60],
                packet_size_pct=[100],
                ul_dl_ratio=4.0,
                flows_per_user=1,
                qos_5qi=7
            ),
            'VoNR': TrafficProfile(
                name='VoNR',
                traffic_share=0.05,
                packet_size_bytes=[60],
                packet_size_pct=[100],
                ul_dl_ratio=1.0,
                flows_per_user=1,
                qos_5qi=1
            )
        }
    
    def load_profiles(self, profile_configs: List[Dict[str, Any]]) -> List[TrafficProfile]:
        profiles = []
        for config in profile_configs:
            if 'name' in config and config['name'] in self.builtin_profiles:
                profile = self.builtin_profiles[config['name']]
                for key, value in config.items():
                    if hasattr(profile, key):
                        setattr(profile, key, value)
                profiles.append(profile)
            else:
                profiles.append(TrafficProfile(**config))
        return profiles
    
    def generate_packet_sizes_vectorized(self, profile: TrafficProfile, num_packets: int) -> np.ndarray:
        """Highly optimized vectorized packet size generation"""
        if num_packets == 0:
            return np.array([], dtype=int)
        
        # Use pre-computed distribution for maximum performance
        dist = profile.packet_distribution
        random_vals = self.rng.random(num_packets)
        
        # Vectorized binary search using searchsorted (much faster than loops)
        indices = np.searchsorted(dist.cumulative_probs, random_vals, side='right')
        indices = np.clip(indices, 0, len(dist.sizes) - 1)
        
        return dist.sizes[indices]
    
    def generate_packet_sizes(self, profile: TrafficProfile, num_packets: int) -> np.ndarray:
        """Backward compatibility wrapper"""
        return self.generate_packet_sizes_vectorized(profile, num_packets)
    
    def calculate_ul_dl_split(self, total_bytes: int, ul_dl_ratio: float) -> Tuple[int, int]:
        ul_bytes = int(total_bytes * ul_dl_ratio / (1 + ul_dl_ratio))
        dl_bytes = total_bytes - ul_bytes
        return ul_bytes, dl_bytes
    
    def apply_burstiness(self, base_packets: int, profile: TrafficProfile, interval_idx: int) -> int:
        """Apply burstiness pattern for realistic traffic variations"""
        if profile.burstiness_factor <= 1.0:
            return base_packets
        
        # Use deterministic but varying burstiness based on interval
        burst_phase = (interval_idx * 0.1) % (2 * np.pi)
        burst_multiplier = 1.0 + (profile.burstiness_factor - 1.0) * (0.5 + 0.5 * np.sin(burst_phase))
        
        return int(base_packets * burst_multiplier)
    
    def generate_traffic_for_interval(self, profile: TrafficProfile, 
                                    num_users: int, 
                                    bandwidth_per_user_mbps: float,
                                    interval_ms: int,
                                    interval_idx: int = 0) -> Dict[str, Any]:
        interval_sec = interval_ms / 1000.0
        total_bytes_per_user = int(bandwidth_per_user_mbps * 1_000_000 * interval_sec / 8)
        total_bytes = int(total_bytes_per_user * num_users * profile.traffic_share)
        
        # Fast packet count estimation using pre-computed average
        dist = profile.packet_distribution
        avg_packet_size = np.sum(dist.sizes * dist.probabilities)
        base_packets = max(1, int(total_bytes / avg_packet_size))
        
        # Apply burstiness for realistic traffic patterns
        total_packets = self.apply_burstiness(base_packets, profile, interval_idx)
        
        # Generate packet sizes only when needed (performance optimization)
        if total_packets > 0:
            packet_sizes = self.generate_packet_sizes_vectorized(profile, total_packets)
            actual_total_bytes = np.sum(packet_sizes, dtype=np.int64)
            
            # Fast statistics computation
            min_size = int(np.min(packet_sizes))
            max_size = int(np.max(packet_sizes))
            avg_size = int(actual_total_bytes / total_packets)
        else:
            actual_total_bytes = 0
            min_size = max_size = avg_size = 0
            packet_sizes = np.array([], dtype=int)
        
        ul_bytes, dl_bytes = self.calculate_ul_dl_split(actual_total_bytes, profile.ul_dl_ratio)
        
        # Vectorized UL/DL packet split
        if actual_total_bytes > 0:
            ul_ratio = ul_bytes / actual_total_bytes
            ul_packets = int(total_packets * ul_ratio)
            dl_packets = total_packets - ul_packets
        else:
            ul_packets = dl_packets = 0
        
        # Fast user/flow calculations
        active_users = min(num_users, max(1, int(num_users * profile.traffic_share)))
        active_flows = active_users * profile.flows_per_user
        
        # Pre-computed throughput
        throughput_ul_mbps = (ul_bytes * 8) / (interval_sec * 1_000_000) if interval_sec > 0 else 0
        throughput_dl_mbps = (dl_bytes * 8) / (interval_sec * 1_000_000) if interval_sec > 0 else 0
        
        return {
            'profile_name': profile.name,
            'total_packets_ul': ul_packets,
            'total_packets_dl': dl_packets,
            'total_bytes_ul': int(ul_bytes),
            'total_bytes_dl': int(dl_bytes),
            'active_users': active_users,
            'active_flows': active_flows,
            'avg_packet_size': avg_size,
            'min_packet_size': min_size,
            'max_packet_size': max_size,
            'throughput_ul_mbps': throughput_ul_mbps,
            'throughput_dl_mbps': throughput_dl_mbps,
            'qos_5qi': profile.qos_5qi,
            'protocol': profile.protocol,
            'priority': profile.priority
        }