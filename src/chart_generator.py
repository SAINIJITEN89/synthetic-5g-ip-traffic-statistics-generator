import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
import json


class ChartGenerator:
    """
    Comprehensive chart generation system for 5G traffic analysis.
    Creates Claude Code-interpretable visualizations with embedded metadata.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Chart configuration
        self.chart_config = config.get('charts', {})
        self.output_dir = Path(config.get('output_file', 'output/stats.csv')).parent / 'charts'
        self.chart_formats = self.chart_config.get('formats', ['png', 'html'])
        self.dpi = self.chart_config.get('dpi', 300)
        self.figsize = self.chart_config.get('figsize', (12, 8))
        
        # Performance optimization settings
        self.max_chart_points = self.chart_config.get('max_points', 10000)  # Max points per chart
        self.sampling_strategy = self.chart_config.get('sampling_strategy', 'smart')  # 'uniform', 'smart', 'time_based'
        self.preserve_peaks = self.chart_config.get('preserve_peaks', True)
        self.time_aggregation = self.chart_config.get('time_aggregation', 'auto')  # 'auto', 'hour', 'minute', 'none'
        
        # Style configuration
        self.setup_styles()
        
    def setup_styles(self):
        """Configure matplotlib and seaborn styles for professional charts"""
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # Custom color palette for 5G profiles
        self.profile_colors = {
            'eMBB': '#2E86AB',     # Blue - consumer services
            'URLLC': '#A23B72',    # Purple - critical applications  
            'mMTC': '#F18F01',     # Orange - IoT/sensors
            'VoNR': '#C73E1D',     # Red - voice
            'total': '#1B1B1E'     # Dark gray - total traffic
        }
        
    def create_output_directory(self):
        """Create output directory for charts"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _optimize_data_for_charting(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame for chart generation using intelligent sampling strategies.
        Reduces data points while preserving important patterns and peaks.
        """
        if len(df) <= self.max_chart_points:
            # No optimization needed for small datasets - return reference, not copy
            return df
        
        self.logger.info(f"Optimizing dataset: {len(df)} points -> target: {self.max_chart_points} points")
        
        if self.sampling_strategy == 'uniform':
            return self._uniform_sampling(df)
        elif self.sampling_strategy == 'smart':
            return self._smart_sampling(df)
        elif self.sampling_strategy == 'time_based':
            return self._time_based_aggregation(df)
        else:
            return self._smart_sampling(df)  # Default fallback
    
    def _uniform_sampling(self, df: pd.DataFrame) -> pd.DataFrame:
        """Simple uniform sampling - fastest but may miss patterns"""
        step = len(df) // self.max_chart_points
        return df.iloc[::step]  # Remove unnecessary .copy()
    
    def _smart_sampling(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Smart sampling that preserves peaks, valleys, and trend changes.
        Combines uniform sampling with peak preservation.
        """
        # Start with uniform sampling base
        uniform_step = len(df) // (self.max_chart_points // 2)  # Use half points for uniform
        uniform_indices = set(range(0, len(df), uniform_step))
        
        # Always include first and last points
        uniform_indices.add(0)
        uniform_indices.add(len(df) - 1)
        
        if self.preserve_peaks:
            # Find peaks and valleys in throughput data
            dl_data = df['total_throughput_dl_mbps'].values
            ul_data = df['total_throughput_ul_mbps'].values
            
            # Detect peaks (above mean + 1.5*std) and valleys (below mean - 1.5*std)
            dl_mean, dl_std = dl_data.mean(), dl_data.std()
            ul_mean, ul_std = ul_data.mean(), ul_data.std()
            
            peak_threshold_dl = dl_mean + 1.5 * dl_std
            valley_threshold_dl = dl_mean - 1.5 * dl_std
            peak_threshold_ul = ul_mean + 1.5 * ul_std
            valley_threshold_ul = ul_mean - 1.5 * ul_std
            
            # Find peak/valley indices
            peak_indices = set(np.where((dl_data > peak_threshold_dl) | (ul_data > peak_threshold_ul))[0])
            valley_indices = set(np.where((dl_data < valley_threshold_dl) | (ul_data < valley_threshold_ul))[0])
            
            # Combine all important indices
            important_indices = uniform_indices | peak_indices | valley_indices
            
            # If still too many points, sample from important indices
            if len(important_indices) > self.max_chart_points:
                important_list = sorted(important_indices)
                step = len(important_list) // self.max_chart_points
                important_indices = set(important_list[::step])
        else:
            important_indices = uniform_indices
        
        # Sort indices and return sampled dataframe  
        selected_indices = sorted(important_indices)[:self.max_chart_points]
        return df.iloc[selected_indices]  # Remove unnecessary .copy()
    
    def _time_based_aggregation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Time-based aggregation - group by time periods and use averages.
        Best for very long time series data.
        """
        # Convert to datetime if not already (avoid copy if possible)
        if 'datetime' not in df.columns:
            df_work = df.copy()  # Only copy when modification needed
            df_work['datetime'] = pd.to_datetime(df_work['timestamp'], unit='s')
        else:
            df_work = df
        
        # Determine aggregation period based on data span
        time_span = df_work['datetime'].max() - df_work['datetime'].min()
        
        if self.time_aggregation == 'auto':
            # Auto-determine aggregation period
            if time_span.days > 30:
                freq = 'H'  # Hourly for monthly+ data
            elif time_span.days > 1:
                freq = '15T'  # 15-minute for multi-day data
            else:
                freq = '5T'  # 5-minute for single day data
        else:
            freq_map = {'hour': 'H', 'minute': 'T', '5min': '5T', '15min': '15T'}
            freq = freq_map.get(self.time_aggregation, 'H')
        
        # Perform time-based aggregation
        df_resampled = df_work.set_index('datetime').resample(freq).agg({
            'timestamp': 'first',
            'total_throughput_dl_mbps': 'mean',
            'total_throughput_ul_mbps': 'mean',
            **{col: 'mean' for col in df_work.columns if col.endswith('_throughput_dl_mbps') and not col.startswith('total')},
            **{col: 'mean' for col in df_work.columns if col.endswith('_throughput_ul_mbps') and not col.startswith('total')},
            **{col: 'mean' for col in df_work.columns if col.endswith('_packets_dl') or col.endswith('_packets_ul')},
            **{col: 'mean' for col in df_work.columns if col.endswith('_bytes_dl') or col.endswith('_bytes_ul')}
        }).dropna().reset_index()
        
        # Limit to max points if still too many
        if len(df_resampled) > self.max_chart_points:
            step = len(df_resampled) // self.max_chart_points
            df_resampled = df_resampled.iloc[::step]
        
        return df_resampled
        
    def generate_time_series_charts(self, df: pd.DataFrame, base_filename: str) -> Dict[str, List[str]]:
        """
        Generate comprehensive time series charts for throughput analysis.
        Returns dict of chart types and their file paths.
        """
        self.create_output_directory()
        generated_charts = {
            'matplotlib': [],
            'plotly': []
        }
        
        # Convert timestamp to datetime for better plotting (in-place when possible)
        if 'datetime' not in df.columns:
            df_plot = df.copy()  # Only copy if datetime conversion needed
            df_plot['datetime'] = pd.to_datetime(df_plot['timestamp'], unit='s')
        else:
            df_plot = df  # Use reference if datetime already exists
        
        # Optimize data for charting performance
        original_size = len(df_plot)
        df_plot = self._optimize_data_for_charting(df_plot)
        optimized_size = len(df_plot)
        
        if original_size != optimized_size:
            self.logger.info(f"Chart data optimized: {original_size} -> {optimized_size} points " +
                           f"({100 * optimized_size / original_size:.1f}% retained)")
        
        # 1. Total UL/DL Throughput Time Series (Matplotlib)
        if 'png' in self.chart_formats or 'svg' in self.chart_formats:
            matplotlib_files = self._create_matplotlib_charts(df_plot, base_filename)
            generated_charts['matplotlib'].extend(matplotlib_files)
        
        # 2. Interactive Time Series (Plotly)
        if 'html' in self.chart_formats:
            plotly_files = self._create_plotly_charts(df_plot, base_filename)
            generated_charts['plotly'].extend(plotly_files)
        
        # 3. Generate chart metadata for Claude analysis
        self._generate_chart_metadata(df_plot, base_filename, generated_charts)
        
        return generated_charts
    
    def _create_matplotlib_charts(self, df: pd.DataFrame, base_filename: str) -> List[str]:
        """Create static matplotlib charts"""
        chart_files = []
        
        # Chart 1: Dual-axis UL/DL Throughput
        fig, ax1 = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Plot DL throughput (primary axis)
        ax1.plot(df['datetime'], df['total_throughput_dl_mbps'], 
                color=self.profile_colors['total'], linewidth=2, 
                label='Downlink Throughput', alpha=0.8)
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Downlink Throughput (Mbps)', color=self.profile_colors['total'])
        ax1.tick_params(axis='y', labelcolor=self.profile_colors['total'])
        ax1.grid(True, alpha=0.3)
        
        # Create secondary axis for UL throughput
        ax2 = ax1.twinx()
        ax2.plot(df['datetime'], df['total_throughput_ul_mbps'], 
                color='red', linewidth=2, label='Uplink Throughput', alpha=0.8)
        ax2.set_ylabel('Uplink Throughput (Mbps)', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # Format datetime axis
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
        ax1.xaxis.set_major_locator(mdates.HourLocator(interval=max(1, len(df)//20)))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        plt.title('5G Network Throughput Time Series\n(Uplink vs Downlink)', 
                 fontsize=14, fontweight='bold', pad=20)
        
        # Add legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.tight_layout()
        
        # Save in requested formats
        for fmt in self.chart_formats:
            if fmt in ['png', 'svg', 'pdf']:
                chart_path = self.output_dir / f"{base_filename}_throughput_timeseries.{fmt}"
                plt.savefig(chart_path, format=fmt, dpi=self.dpi, bbox_inches='tight')
                chart_files.append(str(chart_path))
        
        plt.close()
        
        # Chart 2: Profile Breakdown Stacked Chart
        if self._has_profile_data(df):
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(self.figsize[0], self.figsize[1] * 1.2), 
                                         dpi=self.dpi, sharex=True)
            
            # Get profile columns
            profile_cols_dl = [col for col in df.columns if col.endswith('_throughput_dl_mbps') and not col.startswith('total')]
            profile_cols_ul = [col for col in df.columns if col.endswith('_throughput_ul_mbps') and not col.startswith('total')]
            
            # Downlink stacked chart
            bottom_dl = np.zeros(len(df))
            for col in profile_cols_dl:
                profile_name = col.split('_')[0]
                color = self.profile_colors.get(profile_name, '#666666')
                ax1.fill_between(df['datetime'], bottom_dl, bottom_dl + df[col], 
                               label=f'{profile_name} DL', color=color, alpha=0.7)
                bottom_dl += df[col]
            
            ax1.set_ylabel('Downlink Throughput (Mbps)')
            ax1.set_title('Downlink Throughput by Profile', fontweight='bold')
            ax1.legend(loc='upper left')
            ax1.grid(True, alpha=0.3)
            
            # Uplink stacked chart
            bottom_ul = np.zeros(len(df))
            for col in profile_cols_ul:
                profile_name = col.split('_')[0]
                color = self.profile_colors.get(profile_name, '#666666')
                ax2.fill_between(df['datetime'], bottom_ul, bottom_ul + df[col], 
                               label=f'{profile_name} UL', color=color, alpha=0.7)
                bottom_ul += df[col]
            
            ax2.set_ylabel('Uplink Throughput (Mbps)')
            ax2.set_xlabel('Time')
            ax2.set_title('Uplink Throughput by Profile', fontweight='bold')
            ax2.legend(loc='upper left')
            ax2.grid(True, alpha=0.3)
            
            # Format datetime axis
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
            ax2.xaxis.set_major_locator(mdates.HourLocator(interval=max(1, len(df)//20)))
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
            
            plt.suptitle('5G Traffic Profile Breakdown', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # Save profile breakdown chart
            for fmt in self.chart_formats:
                if fmt in ['png', 'svg', 'pdf']:
                    chart_path = self.output_dir / f"{base_filename}_profile_breakdown.{fmt}"
                    plt.savefig(chart_path, format=fmt, dpi=self.dpi, bbox_inches='tight')
                    chart_files.append(str(chart_path))
            
            plt.close()
        
        return chart_files
    
    def _create_plotly_charts(self, df: pd.DataFrame, base_filename: str) -> List[str]:
        """Create interactive Plotly charts"""
        chart_files = []
        
        # Interactive Time Series Chart
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Total Throughput (UL/DL)', 'Profile Breakdown'),
            vertical_spacing=0.08,
            specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
        )
        
        # Add total throughput traces
        fig.add_trace(
            go.Scatter(x=df['datetime'], y=df['total_throughput_dl_mbps'],
                      name='Downlink Total', line=dict(color='blue', width=3),
                      hovertemplate='Time: %{x}<br>DL Throughput: %{y:.1f} Mbps<extra></extra>'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=df['datetime'], y=df['total_throughput_ul_mbps'],
                      name='Uplink Total', line=dict(color='red', width=3),
                      hovertemplate='Time: %{x}<br>UL Throughput: %{y:.1f} Mbps<extra></extra>'),
            row=1, col=1, secondary_y=True
        )
        
        # Add profile breakdown if available
        if self._has_profile_data(df):
            profile_cols_dl = [col for col in df.columns if col.endswith('_throughput_dl_mbps') and not col.startswith('total')]
            for col in profile_cols_dl:
                profile_name = col.split('_')[0]
                color = self.profile_colors.get(profile_name, '#666666')
                fig.add_trace(
                    go.Scatter(x=df['datetime'], y=df[col],
                              name=f'{profile_name} DL', 
                              line=dict(color=color, width=2),
                              stackgroup='dl',
                              hovertemplate=f'Time: %{{x}}<br>{profile_name} DL: %{{y:.1f}} Mbps<extra></extra>'),
                    row=2, col=1
                )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text='5G Network Throughput Analysis<br><sub>Interactive Time Series with Profile Breakdown</sub>',
                x=0.5,
                font=dict(size=18)
            ),
            height=800,
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Update axes
        fig.update_yaxes(title_text="Downlink Throughput (Mbps)", row=1, col=1)
        fig.update_yaxes(title_text="Uplink Throughput (Mbps)", row=1, col=1, secondary_y=True)
        fig.update_yaxes(title_text="Profile Throughput (Mbps)", row=2, col=1)
        fig.update_xaxes(title_text="Time", row=2, col=1)
        
        # Add Claude-friendly annotations
        annotations = {
            'data_summary': {
                'total_intervals': len(df),
                'duration_hours': (df['datetime'].max() - df['datetime'].min()).total_seconds() / 3600,
                'avg_dl_throughput': float(df['total_throughput_dl_mbps'].mean()),
                'avg_ul_throughput': float(df['total_throughput_ul_mbps'].mean()),
                'peak_dl_throughput': float(df['total_throughput_dl_mbps'].max()),
                'peak_ul_throughput': float(df['total_throughput_ul_mbps'].max())
            },
            'chart_metadata': {
                'chart_type': 'interactive_time_series',
                'profiles_included': self._get_profile_names(df),
                'time_range': {
                    'start': df['datetime'].min().isoformat(),
                    'end': df['datetime'].max().isoformat()
                }
            }
        }
        
        fig.add_annotation(
            x=0.02, y=0.98, xref="paper", yref="paper",
            text=f"Data Summary: {annotations['data_summary']['total_intervals']} intervals, "
                 f"{annotations['data_summary']['duration_hours']:.1f}h duration",
            showarrow=False, font=dict(size=10), bgcolor="rgba(255,255,255,0.8)"
        )
        
        # Save interactive chart
        chart_path = self.output_dir / f"{base_filename}_interactive.html"
        fig.write_html(chart_path, include_plotlyjs='cdn')
        chart_files.append(str(chart_path))
        
        # Save metadata alongside chart
        metadata_path = self.output_dir / f"{base_filename}_chart_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(annotations, f, indent=2, default=str)
        
        return chart_files
    
    def _has_profile_data(self, df: pd.DataFrame) -> bool:
        """Check if DataFrame contains profile-specific throughput data"""
        profile_cols = [col for col in df.columns if col.endswith('_throughput_dl_mbps') and not col.startswith('total')]
        return len(profile_cols) > 0
    
    def _get_profile_names(self, df: pd.DataFrame) -> List[str]:
        """Extract profile names from DataFrame columns"""
        profile_cols = [col for col in df.columns if col.endswith('_throughput_dl_mbps') and not col.startswith('total')]
        return [col.split('_')[0] for col in profile_cols]
    
    def _generate_chart_metadata(self, df: pd.DataFrame, base_filename: str, generated_charts: Dict[str, List[str]]):
        """Generate comprehensive metadata for Claude Code analysis"""
        metadata = {
            'simulation_metadata': {
                'total_intervals': len(df),
                'time_span_hours': (df['datetime'].max() - df['datetime'].min()).total_seconds() / 3600,
                'interval_duration_minutes': df['timestamp'].diff().median() / 60 if len(df) > 1 else 1,
                'data_start_time': df['datetime'].min().isoformat(),
                'data_end_time': df['datetime'].max().isoformat()
            },
            'throughput_analysis': {
                'downlink': {
                    'average_mbps': float(df['total_throughput_dl_mbps'].mean()),
                    'peak_mbps': float(df['total_throughput_dl_mbps'].max()),
                    'minimum_mbps': float(df['total_throughput_dl_mbps'].min()),
                    'std_deviation': float(df['total_throughput_dl_mbps'].std()),
                    'percentile_95': float(df['total_throughput_dl_mbps'].quantile(0.95))
                },
                'uplink': {
                    'average_mbps': float(df['total_throughput_ul_mbps'].mean()),
                    'peak_mbps': float(df['total_throughput_ul_mbps'].max()),
                    'minimum_mbps': float(df['total_throughput_ul_mbps'].min()),
                    'std_deviation': float(df['total_throughput_ul_mbps'].std()),
                    'percentile_95': float(df['total_throughput_ul_mbps'].quantile(0.95))
                },
                'ul_dl_ratio': float(df['total_throughput_ul_mbps'].mean() / df['total_throughput_dl_mbps'].mean())
            },
            'profile_analysis': self._analyze_profiles(df) if self._has_profile_data(df) else {},
            'pattern_indicators': {
                'has_daily_pattern': self._detect_daily_pattern(df),
                'has_weekly_pattern': self._detect_weekly_pattern(df),
                'peak_hours': self._identify_peak_hours(df),
                'traffic_variability': float(df['total_throughput_dl_mbps'].coefficient_of_variation() if hasattr(df['total_throughput_dl_mbps'], 'coefficient_of_variation') else df['total_throughput_dl_mbps'].std() / df['total_throughput_dl_mbps'].mean())
            },
            'generated_charts': generated_charts,
            'claude_analysis_hints': {
                'focus_areas': ['throughput_trends', 'peak_identification', 'profile_distribution', 'pattern_analysis'],
                'improvement_opportunities': ['load_balancing', 'capacity_planning', 'qos_optimization'],
                'chart_interpretation_notes': [
                    'DL throughput typically 3-10x higher than UL in consumer networks',
                    'Time-of-day patterns should show business hours peaks',
                    'Profile breakdown reveals service type distribution',
                    'Sudden spikes may indicate events or system issues'
                ]
            }
        }
        
        # Save comprehensive metadata
        metadata_path = self.output_dir / f"{base_filename}_analysis_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        self.logger.info(f"Generated chart metadata for Claude analysis: {metadata_path}")
    
    def _analyze_profiles(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze profile-specific throughput patterns"""
        profile_analysis = {}
        profile_names = self._get_profile_names(df)
        
        for profile in profile_names:
            dl_col = f"{profile}_throughput_dl_mbps"
            ul_col = f"{profile}_throughput_ul_mbps"
            
            if dl_col in df.columns and ul_col in df.columns:
                profile_analysis[profile] = {
                    'avg_dl_mbps': float(df[dl_col].mean()),
                    'avg_ul_mbps': float(df[ul_col].mean()),
                    'peak_dl_mbps': float(df[dl_col].max()),
                    'peak_ul_mbps': float(df[ul_col].max()),
                    'traffic_share_dl': float(df[dl_col].mean() / df['total_throughput_dl_mbps'].mean()),
                    'traffic_share_ul': float(df[ul_col].mean() / df['total_throughput_ul_mbps'].mean()),
                    'ul_dl_ratio': float(df[ul_col].mean() / df[dl_col].mean()) if df[dl_col].mean() > 0 else 0
                }
        
        return profile_analysis
    
    def _detect_daily_pattern(self, df: pd.DataFrame) -> bool:
        """Detect if data shows daily patterns (simplified heuristic)"""
        if len(df) < 24:  # Need at least 24 data points
            return False
        
        # Simple pattern detection based on coefficient of variation
        hourly_avg = df.groupby(df['datetime'].dt.hour)['total_throughput_dl_mbps'].mean()
        return hourly_avg.std() / hourly_avg.mean() > 0.2  # 20% variation threshold
    
    def _detect_weekly_pattern(self, df: pd.DataFrame) -> bool:
        """Detect if data shows weekly patterns"""
        if len(df) < 7 * 24:  # Need at least a week of hourly data
            return False
        
        daily_avg = df.groupby(df['datetime'].dt.dayofweek)['total_throughput_dl_mbps'].mean()
        return daily_avg.std() / daily_avg.mean() > 0.15  # 15% variation threshold
    
    def _identify_peak_hours(self, df: pd.DataFrame) -> List[int]:
        """Identify peak traffic hours"""
        hourly_avg = df.groupby(df['datetime'].dt.hour)['total_throughput_dl_mbps'].mean()
        threshold = hourly_avg.mean() + hourly_avg.std()
        return hourly_avg[hourly_avg > threshold].index.tolist()
    
    def create_summary_dashboard(self, df: pd.DataFrame, base_filename: str) -> str:
        """Create a comprehensive dashboard combining all visualizations"""
        # Optimize data for dashboard performance (avoid unnecessary copies)
        if 'datetime' not in df.columns:
            df_plot = df.copy()
            df_plot['datetime'] = pd.to_datetime(df_plot['timestamp'], unit='s')
        else:
            df_plot = df
        df_plot = self._optimize_data_for_charting(df_plot)
        
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Total UL/DL Throughput Over Time',
                'Traffic Distribution by Profile', 
                'Hourly Traffic Patterns',
                'Daily Throughput Statistics',
                'Profile Comparison',
                'Peak vs Average Analysis'
            ),
            specs=[[{"colspan": 2}, None], [{}, {}], [{"colspan": 2}, None]],
            vertical_spacing=0.08,
            horizontal_spacing=0.1
        )
        
        # Main time series (row 1, spans both columns)
        fig.add_trace(
            go.Scatter(x=df_plot['datetime'], y=df_plot['total_throughput_dl_mbps'],
                      name='Downlink', line=dict(color='blue', width=2)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df_plot['datetime'], y=df_plot['total_throughput_ul_mbps'],
                      name='Uplink', line=dict(color='red', width=2)),
            row=1, col=1
        )
        
        # Hourly patterns (row 2, col 1)
        if len(df_plot) >= 24:
            hourly_stats = df_plot.groupby(df_plot['datetime'].dt.hour).agg({
                'total_throughput_dl_mbps': ['mean', 'std'],
                'total_throughput_ul_mbps': ['mean', 'std']
            }).round(2)
            
            hours = hourly_stats.index
            fig.add_trace(
                go.Scatter(x=hours, y=hourly_stats[('total_throughput_dl_mbps', 'mean')],
                          name='Avg DL/Hour', line=dict(color='lightblue')),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(x=hours, y=hourly_stats[('total_throughput_ul_mbps', 'mean')],
                          name='Avg UL/Hour', line=dict(color='lightcoral')),
                row=2, col=1
            )
        
        # Profile comparison (row 2, col 2) 
        if self._has_profile_data(df_plot):
            profile_names = self._get_profile_names(df_plot)
            profile_dl_avgs = []
            profile_ul_avgs = []
            
            for profile in profile_names:
                dl_col = f"{profile}_throughput_dl_mbps"
                ul_col = f"{profile}_throughput_ul_mbps"
                if dl_col in df_plot.columns:
                    profile_dl_avgs.append(df_plot[dl_col].mean())
                if ul_col in df_plot.columns:
                    profile_ul_avgs.append(df_plot[ul_col].mean())
            
            fig.add_trace(
                go.Bar(x=profile_names, y=profile_dl_avgs, name='Profile DL Avg', 
                      marker_color='lightblue'),
                row=2, col=2
            )
            fig.add_trace(
                go.Bar(x=profile_names, y=profile_ul_avgs, name='Profile UL Avg',
                      marker_color='lightcoral'),
                row=2, col=2
            )
        
        # Statistics summary (row 3, spans columns)
        stats_data = {
            'Metric': ['Avg DL', 'Peak DL', 'Avg UL', 'Peak UL', 'UL/DL Ratio'],
            'Value': [
                f"{df['total_throughput_dl_mbps'].mean():.1f} Mbps",
                f"{df['total_throughput_dl_mbps'].max():.1f} Mbps", 
                f"{df['total_throughput_ul_mbps'].mean():.1f} Mbps",
                f"{df['total_throughput_ul_mbps'].max():.1f} Mbps",
                f"{df['total_throughput_ul_mbps'].mean() / df['total_throughput_dl_mbps'].mean():.3f}"
            ]
        }
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Metric', 'Value'], fill_color='lightgray'),
                cells=dict(values=[stats_data['Metric'], stats_data['Value']], 
                          fill_color='white')
            ),
            row=3, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text='5G Network Traffic Analysis Dashboard<br><sub>Comprehensive Throughput and Pattern Analysis</sub>',
                x=0.5, font=dict(size=18)
            ),
            height=1200,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        
        # Save dashboard
        dashboard_path = self.output_dir / f"{base_filename}_dashboard.html"
        fig.write_html(dashboard_path, include_plotlyjs='cdn')
        
        self.logger.info(f"Generated comprehensive dashboard: {dashboard_path}")
        return str(dashboard_path)