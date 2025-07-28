#!/usr/bin/env python3

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config_parser import ConfigParser
from src.simulation_engine import SimulationEngine
from src.high_performance_engine import HighPerformanceSimulationEngine
from src.statistics_collector import StatisticsCollector
from src.output_formatter import OutputFormatter


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        config_parser = ConfigParser()
        config = config_parser.get_config()
        
        logger.info("Starting 5G Traffic Statistics Generation")
        logger.info(f"Configuration: {config['num_users']} users, "
                   f"{config['duration_sec']}s duration, "
                   f"{config['interval_ms']}ms intervals")
        
        # Choose simulation engine based on performance requirements
        use_high_performance = config.get('use_high_performance', True)
        
        if use_high_performance:
            logger.info("Using high-performance simulation engine")
            simulation_engine = HighPerformanceSimulationEngine(config)
            
            # Use vectorized mode for maximum performance
            if config.get('use_vectorized', True):
                logger.info("Running vectorized simulation...")
                results_df = simulation_engine.run_vectorized_simulation()
                logger.info(f"Generated {len(results_df)} intervals of data")
                
                # Quick summary calculation from DataFrame
                summary = {
                    'simulation': {
                        'total_duration_sec': len(results_df) * config['interval_ms'] / 1000.0,
                        'total_intervals': len(results_df),
                        'interval_ms': config['interval_ms']
                    },
                    'traffic_totals': {
                        'total_packets_ul': int(results_df['total_packets_ul'].sum()),
                        'total_packets_dl': int(results_df['total_packets_dl'].sum()),
                        'total_bytes_ul': int(results_df['total_bytes_ul'].sum()),
                        'total_bytes_dl': int(results_df['total_bytes_dl'].sum()),
                        'avg_throughput_ul_mbps': float(results_df['total_throughput_ul_mbps'].mean()),
                        'avg_throughput_dl_mbps': float(results_df['total_throughput_dl_mbps'].mean()),
                        'peak_throughput_ul_mbps': float(results_df['total_throughput_ul_mbps'].max()),
                        'peak_throughput_dl_mbps': float(results_df['total_throughput_dl_mbps'].max())
                    }
                }
                
                # Add profile-specific summaries
                for profile in simulation_engine.profiles:
                    profile_name = profile.name
                    if f'{profile_name}_total_packets_ul' in results_df.columns:
                        summary[f'profile_{profile_name}'] = {
                            'total_packets_ul': int(results_df[f'{profile_name}_total_packets_ul'].sum()),
                            'total_packets_dl': int(results_df[f'{profile_name}_total_packets_dl'].sum()),
                            'total_bytes_ul': int(results_df[f'{profile_name}_total_bytes_ul'].sum()),
                            'total_bytes_dl': int(results_df[f'{profile_name}_total_bytes_dl'].sum()),
                            'avg_throughput_ul_mbps': float(results_df[f'{profile_name}_throughput_ul_mbps'].mean()),
                            'avg_throughput_dl_mbps': float(results_df[f'{profile_name}_throughput_dl_mbps'].mean()),
                            'qos_5qi': profile.qos_5qi
                        }
                
                # Add performance stats
                summary['performance'] = simulation_engine.get_performance_stats()
                
            else:
                # Multiprocessing mode
                logger.info("Running multiprocessing simulation...")
                results = simulation_engine.run_simulation()
                logger.info(f"Generated {len(results)} intervals of data")
                
                stats_collector = StatisticsCollector(config)
                for result in results:
                    stats_collector.collect_interval_stats(result)
                
                results_df = stats_collector.to_dataframe()
                summary = stats_collector.calculate_summary_statistics()
                summary['performance'] = simulation_engine.get_performance_stats()
        
        else:
            # Fallback to original engine
            logger.info("Using standard simulation engine")
            simulation_engine = SimulationEngine(config)
            
            if config.get('use_vectorized', True):
                results_df = simulation_engine.run_vectorized_simulation()
                logger.info(f"Generated {len(results_df)} intervals of data")
                
                stats_collector = StatisticsCollector(config)
                summary = {}
                
                for _, row in results_df.iterrows():
                    interval_data = row.to_dict()
                    interval_data['profiles'] = []
                    
                    for profile in simulation_engine.profiles:
                        profile_name = profile.name
                        profile_stats = {
                            'profile_name': profile_name,
                            'total_packets_ul': row.get(f'{profile_name}_total_packets_ul', 0),
                            'total_packets_dl': row.get(f'{profile_name}_total_packets_dl', 0),
                            'total_bytes_ul': row.get(f'{profile_name}_total_bytes_ul', 0),
                            'total_bytes_dl': row.get(f'{profile_name}_total_bytes_dl', 0),
                            'throughput_ul_mbps': row.get(f'{profile_name}_throughput_ul_mbps', 0),
                            'throughput_dl_mbps': row.get(f'{profile_name}_throughput_dl_mbps', 0),
                            'active_users': row.get(f'{profile_name}_active_users', 0),
                            'qos_5qi': profile.qos_5qi
                        }
                        interval_data['profiles'].append(profile_stats)
                    
                    stats_collector.collect_interval_stats(interval_data)
                
                summary = stats_collector.calculate_summary_statistics()
                
            else:
                results = simulation_engine.run_simulation()
                logger.info(f"Generated {len(results)} intervals of data")
                
                stats_collector = StatisticsCollector(config)
                for result in results:
                    stats_collector.collect_interval_stats(result)
                
                results_df = stats_collector.to_dataframe()
                summary = stats_collector.calculate_summary_statistics()
        
        output_formatter = OutputFormatter(config)
        output_files = output_formatter.export_results(results_df, summary)
        
        logger.info("Simulation completed successfully!")
        logger.info("Output files generated:")
        for file_path in output_files:
            logger.info(f"  {file_path}")
        
        if config.get('generate_report', True):
            report_file = output_formatter.write_text_report(summary)
            logger.info(f"  {report_file}")
        
        print(output_formatter.create_report_text(summary))
        
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()