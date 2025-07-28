import yaml
import json
import argparse
from typing import Dict, Any, List, Optional
from pathlib import Path


class ConfigParser:
    def __init__(self):
        self.default_config = {
            'duration_sec': 60,
            'interval_ms': 1000,
            'num_users': 1000,
            'bandwidth_per_user_mbps': 2.0,
            'random_seed': 42,
            'output_file': 'stats.csv',
            'output_format': 'csv',
            'profiles': [
                {
                    'name': 'eMBB',
                    'traffic_share': 0.85,
                    'packet_size_bytes': [1400, 200],
                    'packet_size_pct': [88, 12],
                    'ul_dl_ratio': 0.12,
                    'flows_per_user': 2,
                    'qos_5qi': 9
                }
            ]
        }
    
    def parse_args(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser(description='5G Traffic Statistics Generator')
        parser.add_argument('--config', type=str, help='Path to YAML/JSON config file')
        parser.add_argument('--duration', type=int, help='Simulation duration in seconds')
        parser.add_argument('--interval', type=int, help='Output interval in milliseconds')
        parser.add_argument('--users', type=int, help='Number of users')
        parser.add_argument('--bandwidth', type=float, help='Bandwidth per user in Mbps')
        parser.add_argument('--output', type=str, help='Output file path')
        parser.add_argument('--format', choices=['csv', 'json'], help='Output format')
        parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
        return parser.parse_args()
    
    def load_config_file(self, config_path: str) -> Dict[str, Any]:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(path, 'r') as f:
            if path.suffix.lower() in ['.yaml', '.yml']:
                return yaml.safe_load(f)
            elif path.suffix.lower() == '.json':
                return json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {path.suffix}")
    
    def merge_configs(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        merged = base_config.copy()
        for key, value in override_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self.merge_configs(merged[key], value)
            else:
                merged[key] = value
        return merged
    
    def get_config(self) -> Dict[str, Any]:
        args = self.parse_args()
        config = self.default_config.copy()
        
        if args.config:
            file_config = self.load_config_file(args.config)
            config = self.merge_configs(config, file_config)
        
        cli_overrides = {}
        if args.duration is not None:
            cli_overrides['duration_sec'] = args.duration
        if args.interval is not None:
            cli_overrides['interval_ms'] = args.interval
        if args.users is not None:
            cli_overrides['num_users'] = args.users
        if args.bandwidth is not None:
            cli_overrides['bandwidth_per_user_mbps'] = args.bandwidth
        if args.output is not None:
            cli_overrides['output_file'] = args.output
        if args.format is not None:
            cli_overrides['output_format'] = args.format
        if args.seed is not None:
            cli_overrides['random_seed'] = args.seed
        
        config = self.merge_configs(config, cli_overrides)
        
        self.validate_config(config)
        return config
    
    def validate_config(self, config: Dict[str, Any]):
        required_keys = ['duration_sec', 'interval_ms', 'num_users', 'profiles']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required configuration key: {key}")
        
        if config['duration_sec'] <= 0:
            raise ValueError("Duration must be positive")
        if config['interval_ms'] <= 0:
            raise ValueError("Interval must be positive")
        if config['num_users'] <= 0:
            raise ValueError("Number of users must be positive")
        if not config['profiles']:
            raise ValueError("At least one traffic profile must be defined")