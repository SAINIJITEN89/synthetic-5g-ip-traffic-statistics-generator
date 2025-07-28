import numpy as np
import pandas as pd
from scipy import stats, signal
from scipy.stats import pearsonr
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
import json
from pathlib import Path


class ThroughputAnalyzer:
    """
    Statistical analysis and pattern recognition for 5G traffic throughput data.
    Generates Claude Code-interpretable insights and recommendations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.analysis_config = config.get('analysis', {})
        
        # Analysis parameters
        self.peak_threshold_factor = self.analysis_config.get('peak_threshold', 1.5)  # Peak = mean + 1.5*std
        self.pattern_detection_sensitivity = self.analysis_config.get('pattern_sensitivity', 0.2)
        self.anomaly_threshold = self.analysis_config.get('anomaly_threshold', 2.0)  # 2 standard deviations
        
    def analyze_throughput_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive throughput analysis with pattern recognition.
        Returns structured analysis for Claude Code interpretation.
        """
        # Ensure we have the required columns
        if 'total_throughput_dl_mbps' not in df.columns or 'total_throughput_ul_mbps' not in df.columns:
            raise ValueError("DataFrame must contain 'total_throughput_dl_mbps' and 'total_throughput_ul_mbps' columns")
        
        # Convert timestamp to datetime if needed
        if 'datetime' not in df.columns:
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        
        analysis_results = {
            'basic_statistics': self._calculate_basic_statistics(df),
            'time_patterns': self._analyze_time_patterns(df),
            'peak_analysis': self._analyze_peaks(df),
            'trend_analysis': self._analyze_trends(df),
            'anomaly_detection': self._detect_anomalies(df),
            'profile_analysis': self._analyze_profiles(df),
            'correlation_analysis': self._analyze_correlations(df),
            'performance_insights': self._generate_performance_insights(df),
            'claude_recommendations': self._generate_claude_recommendations(df)
        }
        
        return analysis_results
    
    def _calculate_basic_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive basic statistics"""
        throughput_stats = {}
        
        for direction in ['dl', 'ul']:
            col = f'total_throughput_{direction}_mbps'
            data = df[col]
            
            throughput_stats[direction] = {
                'mean': float(data.mean()),
                'median': float(data.median()),
                'std': float(data.std()),
                'min': float(data.min()),
                'max': float(data.max()),
                'range': float(data.max() - data.min()),
                'coefficient_of_variation': float(data.std() / data.mean() if data.mean() > 0 else 0),
                'percentiles': {
                    '5th': float(data.quantile(0.05)),
                    '25th': float(data.quantile(0.25)),
                    '75th': float(data.quantile(0.75)),
                    '95th': float(data.quantile(0.95)),
                    '99th': float(data.quantile(0.99))
                },
                'skewness': float(stats.skew(data)),
                'kurtosis': float(stats.kurtosis(data))
            }
        
        # Calculate combined statistics
        throughput_stats['combined'] = {
            'total_average_throughput': throughput_stats['dl']['mean'] + throughput_stats['ul']['mean'],
            'ul_dl_ratio': throughput_stats['ul']['mean'] / throughput_stats['dl']['mean'] if throughput_stats['dl']['mean'] > 0 else 0,
            'asymmetry_factor': abs(throughput_stats['dl']['coefficient_of_variation'] - throughput_stats['ul']['coefficient_of_variation']),
            'peak_to_average_ratio_dl': throughput_stats['dl']['max'] / throughput_stats['dl']['mean'] if throughput_stats['dl']['mean'] > 0 else 0,
            'peak_to_average_ratio_ul': throughput_stats['ul']['max'] / throughput_stats['ul']['mean'] if throughput_stats['ul']['mean'] > 0 else 0
        }
        
        return throughput_stats
    
    def _analyze_time_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze temporal patterns in throughput data"""
        patterns = {}
        
        # Hourly patterns
        if len(df) >= 24:  # Need at least 24 data points for hourly analysis
            hourly_dl = df.groupby(df['datetime'].dt.hour)['total_throughput_dl_mbps'].agg(['mean', 'std', 'count'])
            hourly_ul = df.groupby(df['datetime'].dt.hour)['total_throughput_ul_mbps'].agg(['mean', 'std', 'count'])
            
            patterns['hourly'] = {
                'dl_peak_hour': int(hourly_dl['mean'].idxmax()),
                'dl_lowest_hour': int(hourly_dl['mean'].idxmin()),
                'dl_peak_value': float(hourly_dl['mean'].max()),
                'dl_lowest_value': float(hourly_dl['mean'].min()),
                'dl_hourly_variation': float(hourly_dl['mean'].std() / hourly_dl['mean'].mean()),
                
                'ul_peak_hour': int(hourly_ul['mean'].idxmax()),
                'ul_lowest_hour': int(hourly_ul['mean'].idxmin()),
                'ul_peak_value': float(hourly_ul['mean'].max()),
                'ul_lowest_value': float(hourly_ul['mean'].min()),
                'ul_hourly_variation': float(hourly_ul['mean'].std() / hourly_ul['mean'].mean()),
                
                'pattern_strength': float(max(hourly_dl['mean'].std() / hourly_dl['mean'].mean(),
                                            hourly_ul['mean'].std() / hourly_ul['mean'].mean()))
            }
        
        # Daily patterns (if we have more than a week of data)
        if len(df) >= 7 * 24:
            daily_dl = df.groupby(df['datetime'].dt.dayofweek)['total_throughput_dl_mbps'].mean()
            daily_ul = df.groupby(df['datetime'].dt.dayofweek)['total_throughput_ul_mbps'].mean()
            
            weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            patterns['daily'] = {
                'dl_peak_day': weekdays[daily_dl.idxmax()],
                'dl_lowest_day': weekdays[daily_dl.idxmin()],
                'ul_peak_day': weekdays[daily_ul.idxmax()],
                'ul_lowest_day': weekdays[daily_ul.idxmin()],
                'weekend_vs_weekday_dl': float(daily_dl[5:].mean() / daily_dl[:5].mean()),
                'weekend_vs_weekday_ul': float(daily_ul[5:].mean() / daily_ul[:5].mean()),
                'weekly_variation_dl': float(daily_dl.std() / daily_dl.mean()),
                'weekly_variation_ul': float(daily_ul.std() / daily_ul.mean())
            }
        
        # Detect cyclical patterns using autocorrelation
        patterns['cyclical'] = self._detect_cyclical_patterns(df)
        
        return patterns
    
    def _detect_cyclical_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect cyclical patterns using autocorrelation analysis"""
        cyclical = {}
        
        for direction in ['dl', 'ul']:
            col = f'total_throughput_{direction}_mbps'
            data = df[col].values
            
            # Calculate autocorrelation for different lags
            max_lag = min(len(data) // 4, 144)  # Up to 144 intervals (24 hours if 10-min intervals)
            lags = range(1, max_lag)
            autocorr = [abs(np.corrcoef(data[:-lag], data[lag:])[0, 1]) for lag in lags]
            
            # Find peaks in autocorrelation (indicating cyclical patterns)
            peaks, _ = signal.find_peaks(autocorr, height=0.3, distance=10)
            
            cyclical[direction] = {
                'has_strong_cycles': len(peaks) > 0 and max(autocorr) > 0.5,
                'strongest_cycle_lag': int(lags[np.argmax(autocorr)]) if autocorr else 0,
                'cycle_strength': float(max(autocorr)) if autocorr else 0,
                'detected_cycles': [int(lags[peak]) for peak in peaks] if len(peaks) > 0 else []
            }
        
        return cyclical
    
    def _analyze_peaks(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze peak throughput periods and characteristics"""
        peaks = {}
        
        for direction in ['dl', 'ul']:
            col = f'total_throughput_{direction}_mbps'
            data = df[col]
            
            # Define peak threshold
            threshold = data.mean() + self.peak_threshold_factor * data.std()
            peak_mask = data > threshold
            
            if peak_mask.any():
                peak_data = df[peak_mask]
                
                peaks[direction] = {
                    'peak_threshold': float(threshold),
                    'num_peaks': int(peak_mask.sum()),
                    'peak_percentage': float(peak_mask.sum() / len(df) * 100),
                    'avg_peak_value': float(data[peak_mask].mean()),
                    'max_peak_value': float(data[peak_mask].max()),
                    'peak_duration_avg': self._calculate_peak_duration(peak_mask),
                    'peak_times': self._identify_peak_times(peak_data),
                    'time_between_peaks': self._calculate_time_between_peaks(peak_data)
                }
            else:
                peaks[direction] = {
                    'peak_threshold': float(threshold),
                    'num_peaks': 0,
                    'peak_percentage': 0.0,
                    'message': 'No significant peaks detected'
                }
        
        return peaks
    
    def _calculate_peak_duration(self, peak_mask: pd.Series) -> float:
        """Calculate average duration of peak periods"""
        # Find consecutive peak periods
        peak_periods = []
        in_peak = False
        start_idx = 0
        
        for i, is_peak in enumerate(peak_mask):
            if is_peak and not in_peak:
                start_idx = i
                in_peak = True
            elif not is_peak and in_peak:
                peak_periods.append(i - start_idx)
                in_peak = False
        
        # Handle case where data ends during a peak
        if in_peak:
            peak_periods.append(len(peak_mask) - start_idx)
        
        return float(np.mean(peak_periods)) if peak_periods else 0.0
    
    def _identify_peak_times(self, peak_data: pd.DataFrame) -> Dict[str, Any]:
        """Identify when peaks typically occur"""
        if len(peak_data) == 0:
            return {}
        
        peak_hours = peak_data['datetime'].dt.hour.value_counts()
        peak_days = peak_data['datetime'].dt.dayofweek.value_counts()
        
        return {
            'most_common_hour': int(peak_hours.index[0]) if len(peak_hours) > 0 else None,
            'least_common_hour': int(peak_hours.index[-1]) if len(peak_hours) > 0 else None,
            'hourly_distribution': peak_hours.to_dict(),
            'most_common_day': int(peak_days.index[0]) if len(peak_days) > 0 else None,
            'daily_distribution': peak_days.to_dict()
        }
    
    def _calculate_time_between_peaks(self, peak_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate statistics on time between peaks"""
        if len(peak_data) < 2:
            return {'avg_hours': 0, 'min_hours': 0, 'max_hours': 0}
        
        time_diffs = peak_data['datetime'].diff().dropna()
        time_diffs_hours = time_diffs.dt.total_seconds() / 3600
        
        return {
            'avg_hours': float(time_diffs_hours.mean()),
            'min_hours': float(time_diffs_hours.min()),
            'max_hours': float(time_diffs_hours.max()),
            'std_hours': float(time_diffs_hours.std())
        }
    
    def _analyze_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze long-term trends in throughput data"""
        trends = {}
        
        for direction in ['dl', 'ul']:
            col = f'total_throughput_{direction}_mbps'
            data = df[col].values
            time_points = np.arange(len(data))
            
            # Linear trend analysis
            slope, intercept, r_value, p_value, std_err = stats.linregress(time_points, data)
            
            # Mann-Kendall trend test (non-parametric)
            mk_trend, mk_p_value = self._mann_kendall_trend(data)
            
            trends[direction] = {
                'linear_trend': {
                    'slope': float(slope),
                    'r_squared': float(r_value ** 2),
                    'p_value': float(p_value),
                    'significance': 'significant' if p_value < 0.05 else 'not_significant',
                    'direction': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable'
                },
                'mann_kendall': {
                    'trend': mk_trend,
                    'p_value': float(mk_p_value),
                    'significance': 'significant' if mk_p_value < 0.05 else 'not_significant'
                },
                'change_rate_per_hour': float(slope * (3600 / df['timestamp'].diff().median() if len(df) > 1 else 1))
            }
        
        return trends
    
    def _mann_kendall_trend(self, data: np.ndarray) -> Tuple[str, float]:
        """Perform Mann-Kendall trend test"""
        n = len(data)
        if n < 3:
            return 'insufficient_data', 1.0
        
        # Calculate S statistic
        s = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                s += np.sign(data[j] - data[i])
        
        # Calculate variance
        var_s = (n * (n - 1) * (2 * n + 5)) / 18
        
        # Calculate Z statistic
        if s > 0:
            z = (s - 1) / np.sqrt(var_s)
        elif s < 0:
            z = (s + 1) / np.sqrt(var_s)
        else:
            z = 0
        
        # Calculate p-value (two-tailed test)
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        
        # Determine trend direction
        if p_value < 0.05:
            trend = 'increasing' if s > 0 else 'decreasing'
        else:
            trend = 'no_trend'
        
        return trend, p_value
    
    def _detect_anomalies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect anomalies in throughput data"""
        anomalies = {}
        
        for direction in ['dl', 'ul']:
            col = f'total_throughput_{direction}_mbps'
            data = df[col]
            
            # Z-score based anomaly detection
            z_scores = np.abs(stats.zscore(data))
            anomaly_mask = z_scores > self.anomaly_threshold
            
            # IQR based anomaly detection
            q1, q3 = data.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            iqr_anomaly_mask = (data < lower_bound) | (data > upper_bound)
            
            # Combine both methods
            combined_anomaly_mask = anomaly_mask | iqr_anomaly_mask
            
            if combined_anomaly_mask.any():
                anomaly_data = df[combined_anomaly_mask]
                
                anomalies[direction] = {
                    'num_anomalies': int(combined_anomaly_mask.sum()),
                    'anomaly_percentage': float(combined_anomaly_mask.sum() / len(df) * 100),
                    'anomaly_values': {
                        'min': float(data[combined_anomaly_mask].min()),
                        'max': float(data[combined_anomaly_mask].max()),
                        'mean': float(data[combined_anomaly_mask].mean())
                    },
                    'anomaly_times': self._analyze_anomaly_timing(anomaly_data),
                    'severity_distribution': self._categorize_anomaly_severity(data[combined_anomaly_mask], data)
                }
            else:
                anomalies[direction] = {
                    'num_anomalies': 0,
                    'anomaly_percentage': 0.0,
                    'message': 'No significant anomalies detected'
                }
        
        return anomalies
    
    def _analyze_anomaly_timing(self, anomaly_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze when anomalies typically occur"""
        if len(anomaly_data) == 0:
            return {}
        
        return {
            'most_common_hour': int(anomaly_data['datetime'].dt.hour.mode().iloc[0]) if len(anomaly_data) > 0 else None,
            'most_common_day': int(anomaly_data['datetime'].dt.dayofweek.mode().iloc[0]) if len(anomaly_data) > 0 else None,
            'hourly_distribution': anomaly_data['datetime'].dt.hour.value_counts().to_dict(),
            'clustering_detected': self._detect_anomaly_clustering(anomaly_data)
        }
    
    def _detect_anomaly_clustering(self, anomaly_data: pd.DataFrame) -> bool:
        """Detect if anomalies tend to cluster in time"""
        if len(anomaly_data) < 3:
            return False
        
        time_diffs = anomaly_data['datetime'].diff().dropna()
        median_diff = time_diffs.median()
        
        # If more than 50% of anomalies occur within 2x the median time difference, consider them clustered
        close_anomalies = (time_diffs <= 2 * median_diff).sum()
        return close_anomalies / len(time_diffs) > 0.5
    
    def _categorize_anomaly_severity(self, anomaly_values: pd.Series, all_data: pd.Series) -> Dict[str, int]:
        """Categorize anomalies by severity"""
        mean_val = all_data.mean()
        std_val = all_data.std()
        
        severity_counts = {'low': 0, 'medium': 0, 'high': 0, 'extreme': 0}
        
        for value in anomaly_values:
            z_score = abs((value - mean_val) / std_val)
            if z_score < 2.5:
                severity_counts['low'] += 1
            elif z_score < 3.5:
                severity_counts['medium'] += 1
            elif z_score < 5.0:
                severity_counts['high'] += 1
            else:
                severity_counts['extreme'] += 1
        
        return severity_counts
    
    def _analyze_profiles(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze profile-specific patterns if profile data is available"""
        profile_analysis = {}
        
        # Check for profile columns
        profile_cols = [col for col in df.columns if '_throughput_dl_mbps' in col and not col.startswith('total')]
        
        if not profile_cols:
            return {'message': 'No profile-specific data available'}
        
        # Extract profile names
        profile_names = list(set([col.split('_')[0] for col in profile_cols]))
        
        for profile in profile_names:
            dl_col = f"{profile}_throughput_dl_mbps"
            ul_col = f"{profile}_throughput_ul_mbps"
            
            if dl_col in df.columns and ul_col in df.columns:
                dl_data = df[dl_col]
                ul_data = df[ul_col]
                
                profile_analysis[profile] = {
                    'contribution_analysis': {
                        'avg_dl_share': float(dl_data.mean() / df['total_throughput_dl_mbps'].mean()),
                        'avg_ul_share': float(ul_data.mean() / df['total_throughput_ul_mbps'].mean()),
                        'variability_dl': float(dl_data.std() / dl_data.mean() if dl_data.mean() > 0 else 0),
                        'variability_ul': float(ul_data.std() / ul_data.mean() if ul_data.mean() > 0 else 0)
                    },
                    'correlation_with_total': {
                        'dl_correlation': float(pearsonr(dl_data, df['total_throughput_dl_mbps'])[0]),
                        'ul_correlation': float(pearsonr(ul_data, df['total_throughput_ul_mbps'])[0])
                    },
                    'peak_analysis': {
                        'dl_peaks': int((dl_data > dl_data.mean() + 1.5 * dl_data.std()).sum()),
                        'ul_peaks': int((ul_data > ul_data.mean() + 1.5 * ul_data.std()).sum())
                    }
                }
        
        return profile_analysis
    
    def _analyze_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations between different metrics"""
        correlations = {}
        
        # UL vs DL correlation
        ul_dl_corr, ul_dl_p = pearsonr(df['total_throughput_ul_mbps'], df['total_throughput_dl_mbps'])
        correlations['ul_dl_correlation'] = {
            'coefficient': float(ul_dl_corr),
            'p_value': float(ul_dl_p),
            'strength': self._interpret_correlation_strength(abs(ul_dl_corr)),
            'significance': 'significant' if ul_dl_p < 0.05 else 'not_significant'
        }
        
        # Time-based correlations
        df_numeric = df.select_dtypes(include=[np.number])
        if len(df_numeric.columns) > 2:
            corr_matrix = df_numeric.corr()
            
            # Find strongest correlations (excluding self-correlations)
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.5:  # Only include moderate to strong correlations
                        corr_pairs.append({
                            'variables': [col1, col2],
                            'correlation': float(corr_val),
                            'strength': self._interpret_correlation_strength(abs(corr_val))
                        })
            
            correlations['strong_correlations'] = sorted(corr_pairs, 
                                                       key=lambda x: abs(x['correlation']), 
                                                       reverse=True)[:10]  # Top 10
        
        return correlations
    
    def _interpret_correlation_strength(self, correlation: float) -> str:
        """Interpret correlation coefficient strength"""
        if correlation >= 0.9:
            return 'very_strong'
        elif correlation >= 0.7:
            return 'strong'
        elif correlation >= 0.5:
            return 'moderate'
        elif correlation >= 0.3:
            return 'weak'
        else:
            return 'very_weak'
    
    def _generate_performance_insights(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate performance insights for network optimization"""
        insights = {}
        
        # Capacity utilization analysis
        basic_stats = self._calculate_basic_statistics(df)
        
        insights['capacity_analysis'] = {
            'dl_utilization_efficiency': self._calculate_utilization_efficiency(df['total_throughput_dl_mbps']),
            'ul_utilization_efficiency': self._calculate_utilization_efficiency(df['total_throughput_ul_mbps']),
            'asymmetry_level': basic_stats['combined']['ul_dl_ratio'],
            'load_consistency': {
                'dl_consistency': 1 - basic_stats['dl']['coefficient_of_variation'],
                'ul_consistency': 1 - basic_stats['ul']['coefficient_of_variation']
            }
        }
        
        # Performance bottleneck identification
        insights['bottleneck_analysis'] = {
            'potential_dl_congestion_periods': int((df['total_throughput_dl_mbps'] > basic_stats['dl']['percentiles']['95th']).sum()),
            'potential_ul_congestion_periods': int((df['total_throughput_ul_mbps'] > basic_stats['ul']['percentiles']['95th']).sum()),
            'underutilization_periods': {
                'dl_low_usage': int((df['total_throughput_dl_mbps'] < basic_stats['dl']['percentiles']['25th']).sum()),
                'ul_low_usage': int((df['total_throughput_ul_mbps'] < basic_stats['ul']['percentiles']['25th']).sum())
            }
        }
        
        # Quality of service implications
        insights['qos_implications'] = self._analyze_qos_implications(df)
        
        return insights
    
    def _calculate_utilization_efficiency(self, data: pd.Series) -> float:
        """Calculate how efficiently the capacity is utilized"""
        # Efficiency based on how close the average is to the peak, considering variability
        mean_val = data.mean()
        max_val = data.max()
        std_val = data.std()
        
        # Higher efficiency if high average relative to peak, with low variability
        base_efficiency = mean_val / max_val if max_val > 0 else 0
        variability_penalty = std_val / mean_val if mean_val > 0 else 1
        
        return float(base_efficiency * (1 - min(variability_penalty, 0.5)))
    
    def _analyze_qos_implications(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze QoS implications based on traffic patterns"""
        qos_analysis = {}
        
        # Check for profile-specific QoS data
        profile_cols = [col for col in df.columns if '_throughput_dl_mbps' in col and not col.startswith('total')]
        
        if profile_cols:
            profile_names = list(set([col.split('_')[0] for col in profile_cols]))
            
            for profile in profile_names:
                dl_col = f"{profile}_throughput_dl_mbps"
                if dl_col in df.columns:
                    profile_data = df[dl_col]
                    
                    # QoS risk assessment based on variability and peaks
                    variability = profile_data.std() / profile_data.mean() if profile_data.mean() > 0 else 0
                    peak_frequency = (profile_data > profile_data.mean() + 2 * profile_data.std()).sum() / len(profile_data)
                    
                    qos_analysis[profile] = {
                        'variability_risk': 'high' if variability > 0.5 else 'medium' if variability > 0.2 else 'low',
                        'peak_congestion_risk': 'high' if peak_frequency > 0.05 else 'medium' if peak_frequency > 0.02 else 'low',
                        'stability_score': float(1 - variability) * (1 - peak_frequency)
                    }
        
        return qos_analysis
    
    def _generate_claude_recommendations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate Claude Code-interpretable recommendations for system improvements"""
        recommendations = {
            'optimization_opportunities': [],
            'capacity_planning': [],
            'operational_insights': [],
            'data_quality_assessment': [],
            'further_analysis_suggestions': []
        }
        
        basic_stats = self._calculate_basic_statistics(df)
        patterns = self._analyze_time_patterns(df)
        
        # Optimization recommendations
        if basic_stats['combined']['ul_dl_ratio'] < 0.1:
            recommendations['optimization_opportunities'].append({
                'category': 'load_balancing',
                'priority': 'medium',
                'issue': 'Significant UL/DL asymmetry detected',
                'recommendation': 'Consider UL capacity optimization or traffic shaping',
                'expected_impact': 'Improved spectrum efficiency and user experience'
            })
        
        if basic_stats['dl']['coefficient_of_variation'] > 0.5:
            recommendations['optimization_opportunities'].append({
                'category': 'traffic_smoothing',
                'priority': 'high',
                'issue': 'High DL traffic variability detected',
                'recommendation': 'Implement traffic shaping or load balancing algorithms',
                'expected_impact': 'More predictable network performance'
            })
        
        # Capacity planning recommendations
        peak_to_avg_dl = basic_stats['combined']['peak_to_average_ratio_dl']
        if peak_to_avg_dl > 3.0:
            recommendations['capacity_planning'].append({
                'category': 'peak_capacity',
                'priority': 'high',
                'issue': f'High peak-to-average ratio ({peak_to_avg_dl:.1f}:1) in DL',
                'recommendation': 'Consider additional capacity during peak hours or demand response mechanisms',
                'expected_impact': 'Reduced congestion during peak periods'
            })
        
        # Operational insights
        if 'hourly' in patterns and patterns['hourly']['pattern_strength'] > 0.3:
            peak_hour = patterns['hourly']['dl_peak_hour']
            recommendations['operational_insights'].append({
                'category': 'maintenance_scheduling',
                'priority': 'low',
                'insight': f'Strong daily pattern detected with peak at hour {peak_hour}',
                'recommendation': f'Schedule maintenance outside peak hours (avoid hour {peak_hour})',
                'expected_impact': 'Minimized service disruption'
            })
        
        # Data quality assessment
        zero_values_dl = (df['total_throughput_dl_mbps'] == 0).sum()
        if zero_values_dl > 0:
            recommendations['data_quality_assessment'].append({
                'category': 'data_integrity',
                'priority': 'medium',
                'issue': f'{zero_values_dl} intervals with zero DL throughput detected',
                'recommendation': 'Investigate potential measurement issues or actual service outages',
                'expected_impact': 'Improved data reliability for analysis'
            })
        
        # Further analysis suggestions
        recommendations['further_analysis_suggestions'] = [
            {
                'analysis_type': 'geographic_analysis',
                'description': 'Analyze throughput patterns by geographic regions or cell sectors',
                'benefit': 'Identify location-specific optimization opportunities'
            },
            {
                'analysis_type': 'user_behavior_correlation',
                'description': 'Correlate throughput patterns with user demographics and usage patterns',
                'benefit': 'Better understanding of demand drivers'
            },
            {
                'analysis_type': 'seasonal_analysis',
                'description': 'Extend analysis to cover seasonal variations over multiple years',
                'benefit': 'Long-term capacity planning and trend prediction'
            }
        ]
        
        return recommendations
    
    def export_analysis(self, analysis_results: Dict[str, Any], output_path: str):
        """Export analysis results to JSON file for Claude Code consumption"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Add metadata for Claude interpretation
        export_data = {
            'analysis_metadata': {
                'generated_timestamp': datetime.now().isoformat(),
                'analyzer_version': '1.0.0',
                'analysis_type': 'comprehensive_throughput_analysis',
                'claude_interpretation_guide': {
                    'key_metrics': ['basic_statistics', 'peak_analysis', 'performance_insights'],
                    'visualization_suggestions': ['time_series_plots', 'correlation_heatmaps', 'peak_distribution_charts'],
                    'improvement_focus_areas': ['capacity_optimization', 'load_balancing', 'qos_management']
                }
            },
            'analysis_results': analysis_results
        }
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        self.logger.info(f"Analysis results exported to: {output_file}")
        return str(output_file)