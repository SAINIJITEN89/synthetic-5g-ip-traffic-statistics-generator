import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging


class OutputFormatter:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.output_file = config.get('output_file', 'stats.csv')
        self.output_format = config.get('output_format', 'csv')
        self.logger = logging.getLogger(__name__)
    
    def write_dataframe_csv(self, df: pd.DataFrame) -> str:
        output_path = Path(self.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False, float_format='%.3f')
        self.logger.info(f"CSV output written to: {output_path}")
        return str(output_path)
    
    def write_dataframe_json(self, df: pd.DataFrame) -> str:
        output_path = Path(self.output_file)
        if output_path.suffix.lower() == '.csv':
            output_path = output_path.with_suffix('.json')
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'metadata': {
                'total_intervals': len(df),
                'columns': list(df.columns),
                'generated_by': '5G Traffic Statistics Generator'
            },
            'data': df.to_dict('records')
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        self.logger.info(f"JSON output written to: {output_path}")
        return str(output_path)
    
    def write_summary_json(self, summary: Dict[str, Any]) -> str:
        output_path = Path(self.output_file)
        summary_path = output_path.parent / f"{output_path.stem}_summary.json"
        
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"Summary written to: {summary_path}")
        return str(summary_path)
    
    def export_results(self, df: pd.DataFrame, summary: Optional[Dict[str, Any]] = None) -> List[str]:
        output_files = []
        
        if self.output_format.lower() == 'csv':
            csv_file = self.write_dataframe_csv(df)
            output_files.append(csv_file)
        elif self.output_format.lower() == 'json':
            json_file = self.write_dataframe_json(df)
            output_files.append(json_file)
        else:
            csv_file = self.write_dataframe_csv(df)
            json_file = self.write_dataframe_json(df)
            output_files.extend([csv_file, json_file])
        
        if summary:
            summary_file = self.write_summary_json(summary)
            output_files.append(summary_file)
        
        return output_files
    
    def create_report_text(self, summary: Dict[str, Any]) -> str:
        report_lines = [
            "5G Traffic Statistics Generator - Simulation Report",
            "=" * 50,
            ""
        ]
        
        if 'simulation' in summary:
            sim_data = summary['simulation']
            report_lines.extend([
                "Simulation Parameters:",
                f"  Duration: {sim_data.get('total_duration_sec', 0):.1f} seconds",
                f"  Intervals: {sim_data.get('total_intervals', 0)}",
                f"  Interval Size: {sim_data.get('interval_ms', 0)} ms",
                ""
            ])
        
        if 'traffic_totals' in summary:
            traffic = summary['traffic_totals']
            report_lines.extend([
                "Traffic Totals:",
                f"  Uplink Packets: {traffic.get('total_packets_ul', 0):,}",
                f"  Downlink Packets: {traffic.get('total_packets_dl', 0):,}",
                f"  Uplink Bytes: {traffic.get('total_bytes_ul', 0):,}",
                f"  Downlink Bytes: {traffic.get('total_bytes_dl', 0):,}",
                f"  Avg UL Throughput: {traffic.get('avg_throughput_ul_mbps', 0):.2f} Mbps",
                f"  Avg DL Throughput: {traffic.get('avg_throughput_dl_mbps', 0):.2f} Mbps",
                f"  Peak UL Throughput: {traffic.get('peak_throughput_ul_mbps', 0):.2f} Mbps",
                f"  Peak DL Throughput: {traffic.get('peak_throughput_dl_mbps', 0):.2f} Mbps",
                ""
            ])
        
        profile_sections = [key for key in summary.keys() if key.startswith('profile_')]
        if profile_sections:
            report_lines.append("Profile Breakdown:")
            for profile_key in sorted(profile_sections):
                profile_name = profile_key.replace('profile_', '')
                profile_data = summary[profile_key]
                
                report_lines.extend([
                    f"  {profile_name}:",
                    f"    QoS 5QI: {profile_data.get('qos_5qi', 'N/A')}",
                    f"    UL Packets: {profile_data.get('total_packets_ul', 0):,}",
                    f"    DL Packets: {profile_data.get('total_packets_dl', 0):,}",
                    f"    Avg UL Throughput: {profile_data.get('avg_throughput_ul_mbps', 0):.2f} Mbps",
                    f"    Avg DL Throughput: {profile_data.get('avg_throughput_dl_mbps', 0):.2f} Mbps",
                    f"    Avg Active Users: {profile_data.get('avg_active_users', 0):.1f}",
                    ""
                ])
        
        return "\n".join(report_lines)
    
    def write_text_report(self, summary: Dict[str, Any]) -> str:
        output_path = Path(self.output_file)
        report_path = output_path.parent / f"{output_path.stem}_report.txt"
        
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        report_text = self.create_report_text(summary)
        
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        self.logger.info(f"Text report written to: {report_path}")
        return str(report_path)