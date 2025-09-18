# author: carolina ferrari

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from scipy import signal
from scipy.interpolate import interp1d
import warnings
import os
warnings.filterwarnings('ignore')

class DataSynchronizer:
    def __init__(self):
        # Map to actual column names in the datasets
        self.sync_signals = ['wps', 'traction_speed', 'brake_pressure_total', 'G_lateral']
        self.corner_signals = ['wps', 'G_lateral']  # For detecting left/right corners
        self.behavior_signals = ['traction_speed', 'brake_pressure_total']  # For straights and curves
        self.time_column = 'time'
        
    def load_datasets(self, file1_path: str, file2_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load two datasets from CSV files.
        
        Args:
            file1_path: Path to first dataset
            file2_path: Path to second dataset
            
        Returns:
            Tuple of (dataset1, dataset2) DataFrames
        """
        try:
            df1 = pd.read_csv(file1_path)
            df2 = pd.read_csv(file2_path)
            
            # Convert sync-relevant columns to numeric
            sync_cols = ['time', 'wps', 'traction_speed', 'brake_pressure_total', 'G_lateral']
            for col in sync_cols:
                if col in df1.columns:
                    df1[col] = pd.to_numeric(df1[col], errors='coerce')
                if col in df2.columns:
                    df2[col] = pd.to_numeric(df2[col], errors='coerce')
            
            print(f"Loaded dataset 1: {len(df1)} points")
            print(f"Loaded dataset 2: {len(df2)} points")
            
            return df1, df2
        except Exception as e:
            raise Exception(f"Error loading datasets: {e}")
    
    def preprocess_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess signals for synchronization analysis.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Processed DataFrame with additional signal features
        """
        df_processed = df.copy()
        
        # Smooth signals to reduce noise
        for signal in self.sync_signals:
            if signal in df_processed.columns:
                # Apply moving average filter
                window_size = min(5, len(df_processed) // 20)
                if window_size > 1:
                    df_processed[f'{signal}_smooth'] = df_processed[signal].rolling(
                        window=window_size, center=True
                    ).mean()
                else:
                    df_processed[f'{signal}_smooth'] = df_processed[signal]
        
        # Calculate signal derivatives for feature detection
        for signal in self.sync_signals:
            if signal in df_processed.columns:
                df_processed[f'{signal}_diff'] = df_processed[signal].diff()
                df_processed[f'{signal}_abs_diff'] = np.abs(df_processed[f'{signal}_diff'])
        
        # Detect corners using WPS and curvature
        df_processed['corner_detection'] = self._detect_corners(df_processed)
        
        # Detect braking events
        df_processed['braking_event'] = self._detect_braking_events(df_processed)
        
        # Detect speed changes
        df_processed['speed_change'] = self._detect_speed_changes(df_processed)
        
        return df_processed
    
    def _detect_corners(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect corners using WPS and curvature signals.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Series with corner detection flags
        """
        corner_flags = pd.Series(0, index=df.index)
        
        # Use WPS to detect steering changes
        if 'WPS_smooth' in df.columns:
            wps_threshold = df['WPS_smooth'].std() * 0.5
            corner_flags += (np.abs(df['WPS_smooth']) > wps_threshold).astype(int)
        
        # Use lateral G-force to detect turns
        if 'Força_G_lateral_smooth' in df.columns:
            lateral_g_threshold = df['Força_G_lateral_smooth'].std() * 0.3
            corner_flags += (np.abs(df['Força_G_lateral_smooth']) > lateral_g_threshold).astype(int)
        
        return corner_flags
    
    def _detect_braking_events(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect significant braking events.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Series with braking event flags
        """
        if 'Pressão_do_freio_smooth' not in df.columns:
            return pd.Series(0, index=df.index)
        
        brake_threshold = df['Pressão_do_freio_smooth'].quantile(0.7)
        return (df['Pressão_do_freio_smooth'] > brake_threshold).astype(int)
    
    def _detect_speed_changes(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect significant speed changes.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Series with speed change flags
        """
        if 'Velocidade_de_tração_smooth' not in df.columns:
            return pd.Series(0, index=df.index)
        
        speed_diff_threshold = df['Velocidade_de_tração_smooth'].diff().std() * 1.5
        return (np.abs(df['Velocidade_de_tração_smooth'].diff()) > speed_diff_threshold).astype(int)
    
    def find_sync_offset(self, df1: pd.DataFrame, df2: pd.DataFrame, 
                        max_offset: int = None) -> Tuple[int, float]:
        """
        Find the optimal time offset using peak/trough correlation.
        
        Args:
            df1: First dataset (reference)
            df2: Second dataset (to be shifted)
            max_offset: Maximum offset to search (default: 20% of dataset length)
            
        Returns:
            Tuple of (optimal_offset, correlation_score)
        """
        if max_offset is None:
            max_offset = min(len(df1), len(df2)) // 10
        
        print("Detecting peaks and troughs...")
        
        # Detect peaks and troughs for key signals
        df1_features = self._extract_peak_features(df1)
        df2_features = self._extract_peak_features(df2)
        
        print("Finding optimal offset using peak correlation...")
        
        best_offset = 0
        best_score = -1
        
        # Search through possible offsets
        for offset in range(-max_offset, max_offset + 1, 3):  # Smaller step for more precision
            score = self._calculate_peak_correlation(df1_features, df2_features, offset)
            
            if score > best_score:
                best_score = score
                best_offset = offset
        
        # Fine search around best result
        print("Fine-tuning offset...")
        for offset in range(max(-max_offset, best_offset - 5), 
                           min(max_offset, best_offset + 6)):
            if offset != best_offset:
                score = self._calculate_peak_correlation(df1_features, df2_features, offset)
                
                if score > best_score:
                    best_score = score
                    best_offset = offset
        
        print(f"Best sync offset: {best_offset} points")
        print(f"Peak correlation score: {best_score:.3f}")
        
        return best_offset, best_score
    
    def _extract_peak_features(self, df: pd.DataFrame) -> dict:
        """
        Extract peak and trough features from key signals.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with peak/trough information
        """
        features = {}
        
        # Speed peaks (acceleration/deceleration points)
        if 'traction_speed' in df.columns:
            speed_peaks = self._find_peaks_and_troughs(df['traction_speed'])
            features['speed_peaks'] = speed_peaks['peaks']
            features['speed_troughs'] = speed_peaks['troughs']
        
        # Brake pressure peaks (braking events)
        if 'brake_pressure_total' in df.columns:
            brake_peaks = self._find_peaks_and_troughs(df['brake_pressure_total'])
            features['brake_peaks'] = brake_peaks['peaks']
            features['brake_troughs'] = brake_peaks['troughs']
        
        # WPS peaks (steering changes)
        if 'wps' in df.columns:
            wps_peaks = self._find_peaks_and_troughs(df['wps'])
            features['wps_peaks'] = wps_peaks['peaks']
            features['wps_troughs'] = wps_peaks['troughs']
        
        # Lateral G-force peaks (cornering events)
        if 'G_lateral' in df.columns:
            lateral_peaks = self._find_peaks_and_troughs(df['G_lateral'])
            features['lateral_peaks'] = lateral_peaks['peaks']
            features['lateral_troughs'] = lateral_peaks['troughs']
        
        return features
    
    def _find_peaks_and_troughs(self, signal: pd.Series, prominence: float = None) -> dict:
        """
        Find peaks and troughs in a signal.
        
        Args:
            signal: Input signal
            prominence: Minimum prominence for peaks
            
        Returns:
            Dictionary with peaks and troughs indices
        """
        from scipy.signal import find_peaks
        
        if prominence is None:
            prominence = signal.std() * 0.1  # Even lower prominence for more peaks
        
        # Find peaks
        peaks, _ = find_peaks(signal, prominence=prominence, distance=3)
        
        # Find troughs (negative peaks)
        troughs, _ = find_peaks(-signal, prominence=prominence, distance=3)
        
        return {
            'peaks': peaks.tolist(),
            'troughs': troughs.tolist()
        }
    
    def _calculate_peak_correlation(self, features1: dict, features2: dict, offset: int) -> float:
        """
        Calculate correlation score based on peak/trough alignment.
        
        Args:
            features1: Features from first dataset
            features2: Features from second dataset
            offset: Time offset to apply
            
        Returns:
            Correlation score
        """
        total_score = 0.0
        total_weight = 0.0
        
        # Speed peaks correlation (high weight - important for track position)
        if 'speed_peaks' in features1 and 'speed_peaks' in features2:
            speed_score = self._align_peaks(features1['speed_peaks'], features2['speed_peaks'], offset)
            total_score += speed_score * 0.3
            total_weight += 0.3
        
        # Brake peaks correlation (high weight - braking events are distinctive)
        if 'brake_peaks' in features1 and 'brake_peaks' in features2:
            brake_score = self._align_peaks(features1['brake_peaks'], features2['brake_peaks'], offset)
            total_score += brake_score * 0.3
            total_weight += 0.3
        
        # WPS peaks correlation (medium weight - steering changes)
        if 'wps_peaks' in features1 and 'wps_peaks' in features2:
            wps_score = self._align_peaks(features1['wps_peaks'], features2['wps_peaks'], offset)
            total_score += wps_score * 0.2
            total_weight += 0.2
        
        # Lateral G peaks correlation (medium weight - cornering)
        if 'lateral_peaks' in features1 and 'lateral_peaks' in features2:
            lateral_score = self._align_peaks(features1['lateral_peaks'], features2['lateral_peaks'], offset)
            total_score += lateral_score * 0.2
            total_weight += 0.2
        
        if total_weight == 0:
            return 0.0
        
        return total_score / total_weight
    
    def _align_peaks(self, peaks1: list, peaks2: list, offset: int) -> float:
        """
        Calculate alignment score between two sets of peaks with given offset.
        
        Args:
            peaks1: Peaks from first dataset
            peaks2: Peaks from second dataset
            offset: Time offset to apply
            
        Returns:
            Alignment score (0-1)
        """
        if not peaks1 or not peaks2:
            return 0.0
        
        # Apply offset to peaks2
        adjusted_peaks2 = [p + offset for p in peaks2]
        
        # Count matches within tolerance
        tolerance = 20  # Reduced tolerance for more precise matching
        matches = 0
        
        for peak1 in peaks1:
            for peak2 in adjusted_peaks2:
                if abs(peak1 - peak2) <= tolerance:
                    matches += 1
                    break  # Each peak can only match once
        
        # Calculate score based on match ratio
        max_possible_matches = min(len(peaks1), len(peaks2))
        if max_possible_matches == 0:
            return 0.0
        
        return matches / max_possible_matches
    
    def _calculate_curve_correlation(self, df1: pd.DataFrame, df2: pd.DataFrame, 
                                    offset: int) -> float:
        """
        Calculate correlation focusing on curve overlap points and trajectory matching.
        
        Args:
            df1: Reference dataset
            df2: Dataset to be shifted
            offset: Time offset to apply
            
        Returns:
            Enhanced correlation score focusing on curve overlap
        """
        if offset >= 0:
            df1_subset = df1.iloc[offset:]
            df2_subset = df2.iloc[:len(df1_subset)]
        else:
            df2_subset = df2.iloc[-offset:]
            df1_subset = df1.iloc[:len(df2_subset)]
        
        if len(df1_subset) == 0 or len(df2_subset) == 0:
            return 0.0
        
        correlations = []
        weights = []
        
        # WPS correlation (steering angle) - highest weight for curve detection
        if 'WPS_smooth' in df1_subset.columns and 'WPS_smooth' in df2_subset.columns:
            wps1 = df1_subset['WPS_smooth'].dropna().values
            wps2 = df2_subset['WPS_smooth'].dropna().values
            
            # Ensure arrays have same length
            min_len = min(len(wps1), len(wps2))
            if min_len > 10:  # Need minimum data points
                wps1 = wps1[:min_len]
                wps2 = wps2[:min_len]
                wps_corr = np.corrcoef(wps1, wps2)[0, 1]
                if not np.isnan(wps_corr):
                    correlations.append(wps_corr)
                    weights.append(0.4)  # High weight for steering
        
        # Lateral G-force correlation - important for curve dynamics
        if 'Força_G_lateral_smooth' in df1_subset.columns and 'Força_G_lateral_smooth' in df2_subset.columns:
            lateral1 = df1_subset['Força_G_lateral_smooth'].dropna().values
            lateral2 = df2_subset['Força_G_lateral_smooth'].dropna().values
            
            min_len = min(len(lateral1), len(lateral2))
            if min_len > 10:
                lateral1 = lateral1[:min_len]
                lateral2 = lateral2[:min_len]
                lateral_corr = np.corrcoef(lateral1, lateral2)[0, 1]
                if not np.isnan(lateral_corr):
                    correlations.append(lateral_corr)
                    weights.append(0.3)  # High weight for lateral forces
        
        # Speed correlation - moderate weight
        if 'Velocidade_de_tração_smooth' in df1_subset.columns and 'Velocidade_de_tração_smooth' in df2_subset.columns:
            speed1 = df1_subset['Velocidade_de_tração_smooth'].dropna().values
            speed2 = df2_subset['Velocidade_de_tração_smooth'].dropna().values
            
            min_len = min(len(speed1), len(speed2))
            if min_len > 10:
                speed1 = speed1[:min_len]
                speed2 = speed2[:min_len]
                speed_corr = np.corrcoef(speed1, speed2)[0, 1]
                if not np.isnan(speed_corr):
                    correlations.append(speed_corr)
                    weights.append(0.2)  # Moderate weight for speed
        
        # Brake pressure correlation - lower weight
        if 'Pressão_do_freio_smooth' in df1_subset.columns and 'Pressão_do_freio_smooth' in df2_subset.columns:
            brake1 = df1_subset['Pressão_do_freio_smooth'].dropna().values
            brake2 = df2_subset['Pressão_do_freio_smooth'].dropna().values
            
            min_len = min(len(brake1), len(brake2))
            if min_len > 10:
                brake1 = brake1[:min_len]
                brake2 = brake2[:min_len]
                brake_corr = np.corrcoef(brake1, brake2)[0, 1]
                if not np.isnan(brake_corr):
                    correlations.append(brake_corr)
                    weights.append(0.1)  # Lower weight for brake pressure
        
        if not correlations:
            return 0.0
        
        # Weighted average correlation
        weighted_corr = np.average(correlations, weights=weights)
        
        # Simple bonus for curve pattern matching (faster)
        curve_bonus = self._calculate_simple_curve_bonus(df1_subset, df2_subset)
        
        return weighted_corr + curve_bonus
    
    def _calculate_simple_curve_bonus(self, df1: pd.DataFrame, df2: pd.DataFrame) -> float:
        """
        Simple and fast curve pattern bonus calculation.
        
        Args:
            df1: Reference dataset subset
            df2: Dataset subset to compare
            
        Returns:
            Simple bonus score
        """
        if 'WPS_smooth' not in df1.columns or 'WPS_smooth' not in df2.columns:
            return 0.0
        
        # Simple correlation of WPS patterns
        wps1 = df1['WPS_smooth'].dropna().values
        wps2 = df2['WPS_smooth'].dropna().values
        
        min_len = min(len(wps1), len(wps2))
        if min_len > 50:  # Need enough data
            wps1 = wps1[:min_len]
            wps2 = wps2[:min_len]
            
            # Simple correlation
            correlation = np.corrcoef(wps1, wps2)[0, 1]
            if not np.isnan(correlation):
                return correlation * 0.1  # Small bonus
        
        return 0.0
    
    def _calculate_curve_pattern_bonus(self, df1: pd.DataFrame, df2: pd.DataFrame) -> float:
        """
        Calculate bonus score for matching curve patterns (left/right turns).
        
        Args:
            df1: Reference dataset subset
            df2: Dataset subset to compare
            
        Returns:
            Bonus score for curve pattern matching
        """
        if 'WPS_smooth' not in df1.columns or 'WPS_smooth' not in df2.columns:
            return 0.0
        
        # Detect curve directions
        df1_left_turns = (df1['WPS_smooth'] > 0.5).astype(int)
        df1_right_turns = (df1['WPS_smooth'] < -0.5).astype(int)
        
        df2_left_turns = (df2['WPS_smooth'] > 0.5).astype(int)
        df2_right_turns = (df2['WPS_smooth'] < -0.5).astype(int)
        
        # Ensure arrays have same length
        min_len = min(len(df1_left_turns), len(df2_left_turns))
        if min_len > 10:
            df1_left_turns = df1_left_turns[:min_len]
            df2_left_turns = df2_left_turns[:min_len]
            df1_right_turns = df1_right_turns[:min_len]
            df2_right_turns = df2_right_turns[:min_len]
            
            # Calculate pattern similarity
            left_similarity = np.corrcoef(df1_left_turns, df2_left_turns)[0, 1]
            right_similarity = np.corrcoef(df1_right_turns, df2_right_turns)[0, 1]
        else:
            left_similarity = 0.0
            right_similarity = 0.0
        
        if np.isnan(left_similarity):
            left_similarity = 0.0
        if np.isnan(right_similarity):
            right_similarity = 0.0
        
        # Return bonus (max 0.1)
        return (left_similarity + right_similarity) * 0.05
    
    def _calculate_curve_overlap_bonus(self, df1: pd.DataFrame, df2: pd.DataFrame) -> float:
        """
        Calculate bonus score for curve overlap points where both datasets pass through similar track positions.
        
        Args:
            df1: Reference dataset subset
            df2: Dataset subset to compare
            
        Returns:
            Bonus score for curve overlap
        """
        if 'WPS_smooth' not in df1.columns or 'WPS_smooth' not in df2.columns:
            return 0.0
        
        # Find curve points (where steering angle changes significantly)
        df1_curves = self._detect_curve_points(df1['WPS_smooth'])
        df2_curves = self._detect_curve_points(df2['WPS_smooth'])
        
        if len(df1_curves) == 0 or len(df2_curves) == 0:
            return 0.0
        
        # Calculate overlap score based on curve positions
        overlap_score = 0.0
        
        # Check for similar curve patterns
        for i, curve1 in enumerate(df1_curves):
            for j, curve2 in enumerate(df2_curves):
                # Calculate distance between curve points
                time_diff = abs(curve1['time'] - curve2['time'])
                wps_diff = abs(curve1['wps'] - curve2['wps'])
                
                # Bonus for curves that are close in both time and steering angle
                if time_diff < 5.0 and wps_diff < 1.0:  # Within 5 seconds and 1 degree
                    overlap_score += 1.0 / (1.0 + time_diff + wps_diff)
        
        # Normalize bonus (max 0.2)
        max_possible = min(len(df1_curves), len(df2_curves))
        if max_possible > 0:
            return min(overlap_score / max_possible * 0.2, 0.2)
        
        return 0.0
    
    def _detect_curve_points(self, wps_series: pd.Series) -> list:
        """
        Detect significant curve points in WPS data.
        
        Args:
            wps_series: WPS (steering angle) data
            
        Returns:
            List of curve points with time and wps values
        """
        curve_points = []
        
        # Calculate derivative to find steering changes
        wps_diff = wps_series.diff()
        
        # Find significant steering changes (threshold based on data variance)
        threshold = wps_diff.std() * 1.5
        
        for i in range(1, len(wps_series)):
            if abs(wps_diff.iloc[i]) > threshold:
                curve_points.append({
                    'time': i,
                    'wps': wps_series.iloc[i],
                    'change': wps_diff.iloc[i]
                })
        
        return curve_points
    
    def _calculate_correlation(self, df1: pd.DataFrame, df2: pd.DataFrame, 
                              offset: int) -> float:
        """
        Calculate correlation between two datasets with given offset.
        
        Args:
            df1: Reference dataset
            df2: Dataset to be shifted
            offset: Time offset to apply
            
        Returns:
            Correlation score
        """
        if offset >= 0:
            # df2 is shifted forward
            end_idx = min(len(df1) - offset, len(df2))
            if end_idx <= 0:
                return -1
            
            ref_data = df1.iloc[offset:offset + end_idx]
            shift_data = df2.iloc[:end_idx]
        else:
            # df2 is shifted backward
            start_idx = abs(offset)
            end_idx = min(len(df1), len(df2) - start_idx)
            if end_idx <= 0:
                return -1
            
            ref_data = df1.iloc[:end_idx]
            shift_data = df2.iloc[start_idx:start_idx + end_idx]
        
        # Calculate weighted correlation across all sync signals
        total_correlation = 0
        weight_sum = 0
        
        for signal in self.sync_signals:
            if signal in ref_data.columns and signal in shift_data.columns:
                # Normalize signals
                ref_signal = ref_data[signal].fillna(0)
                shift_signal = shift_data[signal].fillna(0)
                
                if ref_signal.std() > 0 and shift_signal.std() > 0:
                    correlation = np.corrcoef(ref_signal, shift_signal)[0, 1]
                    if not np.isnan(correlation):
                        # Weight corner signals more heavily
                        weight = 2.0 if signal in self.corner_signals else 1.0
                        total_correlation += correlation * weight
                        weight_sum += weight
        
        return total_correlation / weight_sum if weight_sum > 0 else -1
    
    def apply_sync_offset(self, df: pd.DataFrame, offset: int) -> pd.DataFrame:
        """
        Apply time offset to synchronize dataset.
        
        Args:
            df: Dataset to shift
            offset: Time offset to apply
            
        Returns:
            Synchronized dataset
        """
        if offset == 0:
            return df.copy()
        
        df_sync = df.copy()
        
        if offset > 0:
            # Shift forward - pad beginning with NaN
            padding = pd.DataFrame(index=range(offset), columns=df.columns)
            df_sync = pd.concat([padding, df_sync], ignore_index=True)
        else:
            # Shift backward - remove beginning points
            df_sync = df_sync.iloc[abs(offset):].reset_index(drop=True)
        
        return df_sync
    
    def visualize_sync_comparison(self, df1: pd.DataFrame, df2: pd.DataFrame, 
                                offset: int, save_path: str = None):
        """
        Create visualization comparing synchronized datasets with superimposed curves.
        
        Args:
            df1: Reference dataset
            df2: Dataset to be synchronized
            offset: Time offset to apply
            save_path: Optional path to save plot
        """
        df2_sync = self.apply_sync_offset(df2, offset)
        
        # Create a comprehensive plot with all signals
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Dataset Synchronization Comparison (Offset: {offset} points)', fontsize=16, fontweight='bold')
        
        # Define colors and styles
        colors = ['#1f77b4', '#ff7f0e']  # Blue and orange
        styles = ['-', '--']
        labels = ['Dataset 1 (Reference)', 'Dataset 2 (Synced)']
        
        # Plot each sync signal
        for i, signal in enumerate(self.sync_signals):
            ax = axes[i // 2, i % 2]
            
            if signal in df1.columns and signal in df2_sync.columns:
                # Plot both datasets with different styles
                ax.plot(df1[self.time_column], df1[signal], color=colors[0], linestyle=styles[0], 
                       label=labels[0], linewidth=2, alpha=0.8)
                ax.plot(df2_sync[self.time_column], df2_sync[signal], color=colors[1], linestyle=styles[1], 
                       label=labels[1], linewidth=2, alpha=0.8)
                
                # Customize plot
                ax.set_xlabel('Time (s)', fontsize=12)
                ax.set_ylabel(self._get_signal_label(signal), fontsize=12)
                ax.set_title(f'{self._get_signal_title(signal)}', fontsize=14, fontweight='bold')
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3)
                
                # Add correlation info
                correlation = df1[signal].corr(df2_sync[signal])
                if not np.isnan(correlation):
                    ax.text(0.02, 0.98, f'Correlation: {correlation:.3f}', 
                           transform=ax.transAxes, fontsize=10, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                           verticalalignment='top')
        
        plt.tight_layout()
        
        # Always save the plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Sync comparison plot saved to: {save_path}")
        else:
            # Save with default name
            default_path = "sync_comparison.png"
            plt.savefig(default_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Sync comparison plot saved to: {default_path}")
        
        # Try to show the plot
        try:
            plt.show(block=False)
            plt.pause(0.1)  # Give time for the plot to render
        except:
            print("Note: Plot display not available in this environment")
            print("Check the saved PNG file for the visualization")
    
    def _get_signal_label(self, signal: str) -> str:
        """Get proper label for signal."""
        labels = {
            'wps': 'Wheel Position (°)',
            'traction_speed': 'Speed (km/h)',
            'brake_pressure_total': 'Brake Pressure (psi)',
            'G_lateral': 'Lateral G-force'
        }
        return labels.get(signal, signal)
    
    def _get_signal_title(self, signal: str) -> str:
        """Get proper title for signal."""
        titles = {
            'wps': 'Wheel Position Sensor',
            'traction_speed': 'Vehicle Speed',
            'brake_pressure_total': 'Brake Pressure',
            'G_lateral': 'Lateral G-force'
        }
        return titles.get(signal, signal)
    
    def create_trajectory_plot(self, df1: pd.DataFrame, df2: pd.DataFrame, 
                             offset: int = 0, save_path: str = "trajectory_comparison.png"):
        """
        Create trajectory visualization showing the path taken by each dataset.
        
        Args:
            df1: First dataset
            df2: Second dataset
            offset: Time offset to apply to df2
            save_path: Path to save the plot
        """
        import matplotlib.pyplot as plt
        
        # Apply offset to df2
        if offset != 0:
            df2_sync = self.apply_sync_offset(df2, offset)
        else:
            df2_sync = df2.copy()
        
        # Create trajectory from WPS (steering angle)
        def create_trajectory(df, color, label, alpha=0.7):
            if 'wps' not in df.columns or 'time' not in df.columns:
                return
            
            # Convert steering angle to trajectory
            # Negative WPS = right turn, Positive WPS = left turn
            x = np.zeros(len(df))
            y = np.zeros(len(df))
            heading = 0
            
            for i in range(1, len(df)):
                # Steering angle affects heading change
                steering_angle = df['wps'].iloc[i]
                dt = df['time'].iloc[i] - df['time'].iloc[i-1]
                
                # Update heading based on steering angle
                heading += steering_angle * dt * 0.1  # Scale factor for visualization
                
                # Update position
                x[i] = x[i-1] + np.cos(heading) * dt * 0.5
                y[i] = y[i-1] + np.sin(heading) * dt * 0.5
            
            return x, y
        
        # Create trajectories
        result1 = create_trajectory(df1, 'blue', 'Dataset 1', 0.8)
        result2 = create_trajectory(df2_sync, 'red', 'Dataset 2 (Synced)', 0.8)
        
        x1, y1 = result1 if result1 is not None else (None, None)
        x2, y2 = result2 if result2 is not None else (None, None)
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Plot 1: Trajectory comparison
        if x1 is not None and y1 is not None:
            ax1.plot(x1, y1, 'b-', linewidth=2, alpha=0.8, label='Dataset 1')
            ax1.scatter(x1[0], y1[0], color='blue', s=100, marker='o', label='Start 1')
            ax1.scatter(x1[-1], y1[-1], color='blue', s=100, marker='s', label='End 1')
        
        if x2 is not None and y2 is not None:
            ax1.plot(x2, y2, 'r--', linewidth=2, alpha=0.8, label='Dataset 2 (Synced)')
            ax1.scatter(x2[0], y2[0], color='red', s=100, marker='o', label='Start 2')
            ax1.scatter(x2[-1], y2[-1], color='red', s=100, marker='s', label='End 2')
        
        ax1.set_xlabel('X Position (relative)', fontsize=12)
        ax1.set_ylabel('Y Position (relative)', fontsize=12)
        ax1.set_title('Trajectory Comparison', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
        
        # Plot 2: WPS comparison over time
        if 'wps' in df1.columns and 'wps' in df2_sync.columns:
            ax2.plot(df1['time'], df1['wps'], 'b-', linewidth=2, alpha=0.8, label='Dataset 1')
            ax2.plot(df2_sync['time'], df2_sync['wps'], 'r--', linewidth=2, alpha=0.8, label='Dataset 2 (Synced)')
            
            # Add horizontal lines for turn detection
            ax2.axhline(y=0.5, color='green', linestyle=':', alpha=0.5, label='Left Turn Threshold')
            ax2.axhline(y=-0.5, color='orange', linestyle=':', alpha=0.5, label='Right Turn Threshold')
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            ax2.set_xlabel('Time (s)', fontsize=12)
            ax2.set_ylabel('WPS (steering angle)', fontsize=12)
            ax2.set_title('Steering Angle Comparison', fontsize=14, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Trajectory plot saved to: {save_path}")
        
        try:
            plt.show(block=False)
            plt.pause(0.1)
        except:
            print("Note: Plot display not available in this environment")
            print("Check the saved PNG file for the visualization")
    
    def manual_sync_adjustment(self, df1: pd.DataFrame, df2: pd.DataFrame) -> int:
        """
        Interactive manual synchronization adjustment.
        
        Args:
            df1: Reference dataset
            df2: Dataset to synchronize
            
        Returns:
            Final offset value
        """
        print("\n=== MANUAL SYNCHRONIZATION ===")
        print("Use the visualization to identify track features and adjust offset.")
        print("Commands:")
        print("  'offset <number>' - Set specific offset")
        print("  'shift <number>' - Shift by amount")
        print("  'plot' - Show current comparison")
        print("  'done' - Accept current offset")
        print("  'quit' - Exit without saving")
        
        current_offset = 0
        
        while True:
            command = input(f"\nCurrent offset: {current_offset} | Enter command: ").strip().lower()
            
            if command == 'done':
                print(f"Final offset: {current_offset}")
                return current_offset
            
            elif command == 'quit':
                print("Exiting without saving.")
                return 0
            
            elif command == 'plot':
                self.visualize_sync_comparison(df1, df2, current_offset)
            
            elif command.startswith('offset '):
                try:
                    new_offset = int(command.split()[1])
                    current_offset = new_offset
                    print(f"Offset set to: {current_offset}")
                except (ValueError, IndexError):
                    print("Invalid offset value. Use: offset <number>")
            
            elif command.startswith('shift '):
                try:
                    shift_amount = int(command.split()[1])
                    current_offset += shift_amount
                    print(f"Offset shifted by {shift_amount} to: {current_offset}")
                except (ValueError, IndexError):
                    print("Invalid shift value. Use: shift <number>")
            
            else:
                print("Unknown command. Available: offset, shift, plot, done, quit")
    
    def auto_sync_datasets_by_disturbances(self, file1_path: str, file2_path: str, 
                                         max_offset: float = 200.0, step: float = 5.0, 
                                         show_plot: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
        """
        Automatically synchronize datasets by detecting and aligning first disturbances.
        
        Args:
            file1_path: Path to reference dataset
            file2_path: Path to dataset to synchronize
            max_offset: Maximum time offset to test (seconds)
            step: Step size for offset testing (seconds)
            show_plot: Whether to display the comparison plot
            
        Returns:
            Tuple of (reference_dataset, synchronized_dataset, optimal_time_offset)
        """
        print("=== AUTOMATIC DISTURBANCE-BASED SYNCHRONIZATION ===")
        
        # Load datasets
        df1, df2 = self.load_datasets(file1_path, file2_path)
        
        # Find optimal offset using disturbance detection
        optimal_offset = self._find_optimal_offset_by_disturbances(df1, df2, max_offset, step)
        
        # Apply optimal offset
        df2_synced = df2.copy()
        df2_synced[self.time_column] = df2[self.time_column] + optimal_offset
        
        print(f"Automatically detected optimal offset: {optimal_offset} seconds")
        print(f"Dataset 1: {len(df1)} points")
        print(f"Dataset 2 (synchronized): {len(df2_synced)} points")
        
        # Calculate overlap statistics
        time1_min, time1_max = df1[self.time_column].min(), df1[self.time_column].max()
        time2_min, time2_max = df2_synced[self.time_column].min(), df2_synced[self.time_column].max()
        
        common_start = max(time1_min, time2_min)
        common_end = min(time1_max, time2_max)
        overlap_duration = common_end - common_start if common_start < common_end else 0
        
        print(f"Overlap duration: {overlap_duration:.1f} seconds")
        print(f"Overlap range: {common_start:.1f}s to {common_end:.1f}s")
        
        # Show comparison plot
        if show_plot:
            print("\nGenerating automatic synchronization comparison plot...")
            self.visualize_time_shift_comparison(df1, df2_synced, optimal_offset, "auto_sync_comparison.png")
        
        return df1, df2_synced, optimal_offset
    
    def _find_optimal_offset_by_disturbances(self, df1: pd.DataFrame, df2: pd.DataFrame, 
                                          max_offset: float, step: float) -> float:
        """
        Find optimal time offset by detecting and aligning first disturbances.
        """
        print(f"Testing offsets from 0 to {max_offset}s (step: {step}s)")
        print("Detecting first disturbances for alignment...")
        
        # Detect disturbances in reference dataset
        ref_disturbances = {}
        for signal in self.sync_signals:
            disturbances = self._detect_first_disturbances(df1, signal)
            ref_disturbances[signal] = disturbances
            print(f"  {signal}: {len(disturbances)} disturbances detected")
        
        best_offset = 0
        best_score = 0
        
        # Test different offsets
        for offset in np.arange(0, max_offset + step, step):
            # Apply offset to df2
            df2_shifted = df2.copy()
            df2_shifted[self.time_column] = df2[self.time_column] + offset
            
            # Detect disturbances in shifted dataset
            shifted_disturbances = {}
            for signal in self.sync_signals:
                disturbances = self._detect_first_disturbances(df2_shifted, signal)
                shifted_disturbances[signal] = disturbances
            
            # Calculate alignment score
            alignment_score = self._calculate_disturbance_alignment_score(
                ref_disturbances, shifted_disturbances
            )
            
            print(f"Offset {offset:3.0f}s: Alignment Score = {alignment_score:.4f}")
            
            if alignment_score > best_score:
                best_score = alignment_score
                best_offset = offset
        
        print(f"Best offset: {best_offset}s (Score: {best_score:.4f})")
        return best_offset
    
    def _detect_first_disturbances(self, df: pd.DataFrame, signal_name: str, 
                                  window_size: int = 20, threshold_factor: float = 2.0) -> List[Dict]:
        """
        Detect the first significant disturbances in a signal.
        """
        if signal_name not in df.columns:
            return []
        
        signal_data = df[signal_name].dropna()
        if len(signal_data) < window_size:
            return []
        
        # Use derivative to find rapid changes
        derivative = np.gradient(signal_data)
        smoothed_derivative = pd.Series(derivative).rolling(window=window_size).mean()
        
        # Find peaks in derivative (rapid changes)
        peaks, properties = signal.find_peaks(np.abs(smoothed_derivative), 
                                            height=np.std(smoothed_derivative) * threshold_factor,
                                            distance=window_size)
        
        disturbances = []
        for peak in peaks[:5]:  # First 5 disturbances
            disturbances.append({
                'index': peak,
                'time': df[self.time_column].iloc[peak],
                'value': signal_data.iloc[peak],
                'derivative': smoothed_derivative.iloc[peak]
            })
        
        return disturbances
    
    def _calculate_disturbance_alignment_score(self, ref_disturbances: Dict, 
                                             shifted_disturbances: Dict) -> float:
        """
        Calculate alignment score based on how well disturbances align.
        """
        total_score = 0
        signal_count = 0
        
        for signal in self.sync_signals:
            ref_dists = ref_disturbances.get(signal, [])
            shift_dists = shifted_disturbances.get(signal, [])
            
            if not ref_dists or not shift_dists:
                continue
            
            signal_count += 1
            
            # Compare timing of first few disturbances
            min_dists = min(len(ref_dists), len(shift_dists), 3)  # Compare first 3
            
            timing_scores = []
            for i in range(min_dists):
                ref_time = ref_dists[i]['time']
                shift_time = shift_dists[i]['time']
                
                # Calculate timing difference
                time_diff = abs(ref_time - shift_time)
                
                # Score based on timing alignment (closer is better)
                timing_score = np.exp(-time_diff / 10.0)  # 10s decay constant
                timing_scores.append(timing_score)
            
            # Average timing score for this signal
            if timing_scores:
                signal_score = np.mean(timing_scores)
                total_score += signal_score
        
        # Return average score across all signals
        return total_score / signal_count if signal_count > 0 else 0

    def sync_datasets_time_shift(self, file1_path: str, file2_path: str, 
                               time_offset: float = 0.0, show_plot: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
        """
        Synchronize datasets using time-shift method (shifts time axis instead of data points).
        
        Args:
            file1_path: Path to reference dataset
            file2_path: Path to dataset to synchronize
            time_offset: Time offset in seconds to apply to dataset 2
            show_plot: Whether to display the comparison plot
            
        Returns:
            Tuple of (reference_dataset, time_shifted_dataset, time_offset_applied)
        """
        print("=== TIME-SHIFT SYNCHRONIZATION ===")
        
        # Load datasets
        df1, df2 = self.load_datasets(file1_path, file2_path)
        
        # Apply time offset to dataset 2
        df2_shifted = df2.copy()
        df2_shifted[self.time_column] = df2[self.time_column] + time_offset
        
        print(f"Applied time offset: {time_offset} seconds")
        print(f"Dataset 1: {len(df1)} points")
        print(f"Dataset 2 (time-shifted): {len(df2_shifted)} points")
        
        # Calculate overlap statistics
        time1_min, time1_max = df1[self.time_column].min(), df1[self.time_column].max()
        time2_min, time2_max = df2_shifted[self.time_column].min(), df2_shifted[self.time_column].max()
        
        common_start = max(time1_min, time2_min)
        common_end = min(time1_max, time2_max)
        overlap_duration = common_end - common_start if common_start < common_end else 0
        
        print(f"Overlap duration: {overlap_duration:.1f} seconds")
        print(f"Overlap range: {common_start:.1f}s to {common_end:.1f}s")
        
        # Show comparison plot
        if show_plot:
            print("\nGenerating time-shift comparison plot...")
            self.visualize_time_shift_comparison(df1, df2_shifted, time_offset, "time_shift_comparison.png")
        
        return df1, df2_shifted, time_offset
    
    def visualize_time_shift_comparison(self, df1: pd.DataFrame, df2_shifted: pd.DataFrame, 
                                      time_offset: float, save_path: str = None):
        """
        Create visualization comparing time-shifted datasets.
        
        Args:
            df1: Reference dataset
            df2_shifted: Time-shifted dataset
            time_offset: Time offset applied
            save_path: Optional path to save plot
        """
        # Create a comprehensive plot with all signals
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Time-Shift Synchronization Comparison (Offset: {time_offset}s)', 
                    fontsize=16, fontweight='bold')
        
        # Define colors and styles
        colors = ['#1f77b4', '#ff7f0e']  # Blue and orange
        styles = ['-', '--']
        labels = ['Dataset 1 (Reference)', 'Dataset 2 (Time-Shifted)']
        
        # Plot each sync signal
        for i, signal in enumerate(self.sync_signals):
            ax = axes[i // 2, i % 2]
            
            if signal in df1.columns and signal in df2_shifted.columns:
                # Plot both datasets with different styles
                ax.plot(df1[self.time_column], df1[signal], color=colors[0], linestyle=styles[0], 
                       label=labels[0], linewidth=2, alpha=0.8)
                ax.plot(df2_shifted[self.time_column], df2_shifted[signal], color=colors[1], linestyle=styles[1], 
                       label=labels[1], linewidth=2, alpha=0.8)
                
                # Customize plot
                ax.set_xlabel('Time (s)', fontsize=12)
                ax.set_ylabel(self._get_signal_label(signal), fontsize=12)
                ax.set_title(f'{self._get_signal_title(signal)}', fontsize=14, fontweight='bold')
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3)
                
                # Add correlation info
                correlation = df1[signal].corr(df2_shifted[signal])
                if not np.isnan(correlation):
                    ax.text(0.02, 0.98, f'Correlation: {correlation:.3f}', 
                           transform=ax.transAxes, fontsize=10, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                           verticalalignment='top')
        
        plt.tight_layout()
        
        # Always save the plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Time-shift comparison plot saved to: {save_path}")
        else:
            # Save with default name
            default_path = "time_shift_comparison.png"
            plt.savefig(default_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Time-shift comparison plot saved to: {default_path}")
        
        # Try to show the plot
        try:
            plt.show(block=False)
            plt.pause(0.1)  # Give time for the plot to render
        except:
            print("Note: Plot display not available in this environment")
            print("Check the saved PNG file for the visualization")

    def sync_datasets(self, file1_path: str, file2_path: str, 
                     manual_mode: bool = False, show_plot: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
        """
        Main synchronization function.
        
        Args:
            file1_path: Path to reference dataset
            file2_path: Path to dataset to synchronize
            manual_mode: Whether to use manual adjustment
            show_plot: Whether to display the comparison plot
            
        Returns:
            Tuple of (synced_dataset1, synced_dataset2, offset_applied)
        """
        print("=== DATASET SYNCHRONIZATION ===")
        
        # Load datasets
        df1, df2 = self.load_datasets(file1_path, file2_path)
        
        # Preprocess signals
        print("Preprocessing signals...")
        df1_processed = self.preprocess_signals(df1)
        df2_processed = self.preprocess_signals(df2)
        
        # Find automatic sync offset
        print("Finding optimal sync offset...")
        auto_offset, correlation = self.find_sync_offset(df1_processed, df2_processed)
        
        if manual_mode:
            print(f"Automatic offset: {auto_offset} (correlation: {correlation:.3f})")
            final_offset = self.manual_sync_adjustment(df1, df2)
        else:
            final_offset = auto_offset
        
        # Apply synchronization
        df2_synced = self.apply_sync_offset(df2, final_offset)
        
        print(f"\nSynchronization complete!")
        print(f"Applied offset: {final_offset} points")
        print(f"Dataset 1: {len(df1)} points")
        print(f"Dataset 2 (synced): {len(df2_synced)} points")
        
        # Show comparison plot
        if show_plot:
            print("\nGenerating comparison plot...")
            self.visualize_sync_comparison(df1, df2_synced, final_offset, "sync_comparison.png")
            
            print("Generating trajectory plot...")
            self.create_trajectory_plot(df1, df2, final_offset, "trajectory_comparison.png")
        
        return df1, df2_synced, final_offset

# Example usage function
def sync_two_datasets(file1_path: str, file2_path: str, 
                     output_path: str = None, manual_mode: bool = False, 
                     show_plot: bool = True):
    """
    Convenience function to sync two datasets.
    
    Args:
        file1_path: Path to reference dataset
        file2_path: Path to dataset to synchronize
        output_path: Optional path to save synced dataset
        manual_mode: Whether to use manual adjustment
        show_plot: Whether to display the comparison plot
    """
    synchronizer = DataSynchronizer()
    
    df1_synced, df2_synced, offset = synchronizer.sync_datasets(
        file1_path, file2_path, manual_mode, show_plot
    )
    
    if output_path:
        # Save synced dataset
        df2_synced.to_csv(output_path, index=False)
        print(f"Synced dataset saved to: {output_path}")
    
    return df1_synced, df2_synced, offset

# Automatic synchronization convenience function
def auto_sync_two_datasets(file1_path: str, file2_path: str, 
                          max_offset: float = 200.0, step: float = 5.0,
                          output_path: str = None, show_plot: bool = True):
    """
    Convenience function to automatically sync two datasets by detecting first disturbances.
    
    Args:
        file1_path: Path to reference dataset
        file2_path: Path to dataset to synchronize
        max_offset: Maximum time offset to test (seconds)
        step: Step size for offset testing (seconds)
        output_path: Optional path to save synchronized dataset
        show_plot: Whether to display the comparison plot
    """
    synchronizer = DataSynchronizer()
    
    df1_synced, df2_synced, offset = synchronizer.auto_sync_datasets_by_disturbances(
        file1_path, file2_path, max_offset, step, show_plot
    )
    
    if output_path:
        # Save synchronized dataset
        df2_synced.to_csv(output_path, index=False)
        print(f"Synchronized dataset saved to: {output_path}")
    
    return df1_synced, df2_synced, offset

# Time-shift synchronization convenience function
def sync_two_datasets_time_shift(file1_path: str, file2_path: str, 
                                time_offset: float = 0.0, output_path: str = None, 
                                show_plot: bool = True):
    """
    Convenience function to sync two datasets using time-shift method.
    
    Args:
        file1_path: Path to reference dataset
        file2_path: Path to dataset to synchronize
        time_offset: Time offset in seconds to apply to dataset 2
        output_path: Optional path to save time-shifted dataset
        show_plot: Whether to display the comparison plot
    """
    synchronizer = DataSynchronizer()
    
    df1_synced, df2_synced, offset = synchronizer.sync_datasets_time_shift(
        file1_path, file2_path, time_offset, show_plot
    )
    
    if output_path:
        # Save time-shifted dataset
        df2_synced.to_csv(output_path, index=False)
        print(f"Time-shifted dataset saved to: {output_path}")
    
    return df1_synced, df2_synced, offset

def get_file_paths():
    """Get file paths from user input."""
    print("=" * 60)
    print("DATA SYNCHRONIZATION TOOL")
    print("=" * 60)
    print("Please provide the paths to your CSV files:")
    print()
    
    # Get first file path
    while True:
        file1_path = input("Enter path to reference dataset (Dataset 1): ").strip()
        if file1_path:
            # Remove quotes if user included them
            file1_path = file1_path.strip('"\'')
            if os.path.exists(file1_path):
                print(f"✅ Found: {file1_path}")
                break
            else:
                print(f"❌ File not found: {file1_path}")
                print("Please check the path and try again.")
        else:
            print("Please enter a valid file path.")
    
    print()
    
    # Get second file path
    while True:
        file2_path = input("Enter path to dataset to synchronize (Dataset 2): ").strip()
        if file2_path:
            # Remove quotes if user included them
            file2_path = file2_path.strip('"\'')
            if os.path.exists(file2_path):
                print(f"✅ Found: {file2_path}")
                break
            else:
                print(f"❌ File not found: {file2_path}")
                print("Please check the path and try again.")
        else:
            print("Please enter a valid file path.")
    
    return file1_path, file2_path

def choose_sync_method():
    """Let user choose synchronization method."""
    print("\n" + "=" * 60)
    print("CHOOSE SYNCHRONIZATION METHOD")
    print("=" * 60)
    print("1. Automatic synchronization (recommended)")
    print("   - Automatically detects optimal time offset")
    print("   - Focuses on first disturbances")
    print("   - No manual input required")
    print()
    print("2. Manual time-shift synchronization")
    print("   - Specify exact time offset")
    print("   - Good for known offsets")
    print()
    print("3. Original synchronization method")
    print("   - Uses peak/trough correlation")
    print("   - May require manual adjustment")
    print()
    
    while True:
        choice = input("Enter your choice (1, 2, or 3): ").strip()
        if choice in ['1', '2', '3']:
            return int(choice)
        else:
            print("Please enter 1, 2, or 3.")

def get_manual_offset():
    """Get manual time offset from user."""
    print("\n" + "=" * 60)
    print("MANUAL TIME OFFSET")
    print("=" * 60)
    print("Enter the time offset in seconds.")
    print("Positive values shift Dataset 2 forward in time.")
    print("Negative values shift Dataset 2 backward in time.")
    print("Example: 55.0 (for 55 seconds forward)")
    print()
    
    while True:
        try:
            offset_input = input("Enter time offset (seconds): ").strip()
            offset = float(offset_input)
            print(f"✅ Time offset set to: {offset} seconds")
            return offset
        except ValueError:
            print("❌ Invalid input. Please enter a number (e.g., 55.0)")

def main_interactive():
    """Main interactive function."""
    try:
        # Get file paths
        file1_path, file2_path = get_file_paths()
        
        # Choose synchronization method
        method = choose_sync_method()
        
        print("\n" + "=" * 60)
        print("STARTING SYNCHRONIZATION")
        print("=" * 60)
        
        if method == 1:
            # Automatic synchronization
            print("🤖 Using automatic synchronization...")
            df1, df2_synced, optimal_offset = auto_sync_two_datasets(
                file1_path, file2_path, 
                max_offset=200.0, 
                step=5.0, 
                show_plot=True
            )
            print(f"\n🎯 AUTOMATIC SYNCHRONIZATION COMPLETE!")
            print(f"Optimal offset detected: {optimal_offset} seconds")
            
        elif method == 2:
            # Manual time-shift synchronization
            offset = get_manual_offset()
            print(f"🔧 Using manual time-shift synchronization with offset: {offset}s...")
            df1, df2_synced, applied_offset = sync_two_datasets_time_shift(
                file1_path, file2_path, 
                time_offset=offset, 
                show_plot=True
            )
            print(f"\n🎯 MANUAL SYNCHRONIZATION COMPLETE!")
            print(f"Applied offset: {applied_offset} seconds")
            
        else:
            # Original synchronization method
            print("🔧 Using original synchronization method...")
            df1, df2_synced, offset = sync_two_datasets(
                file1_path, file2_path, 
                manual_mode=False, 
                show_plot=True
            )
            print(f"\n🎯 ORIGINAL SYNCHRONIZATION COMPLETE!")
            print(f"Data point offset: {offset}")
        
        # Ask if user wants to save the synchronized dataset
        print("\n" + "=" * 60)
        print("SAVE SYNCHRONIZED DATASET")
        print("=" * 60)
        save_choice = input("Do you want to save the synchronized Dataset 2? (y/n): ").strip().lower()
        
        if save_choice in ['y', 'yes']:
            output_path = input("Enter output file path (e.g., 'synchronized_dataset.csv'): ").strip()
            if output_path:
                df2_synced.to_csv(output_path, index=False)
                print(f"✅ Synchronized dataset saved to: {output_path}")
            else:
                print("❌ No output path provided. Dataset not saved.")
        else:
            print("Dataset not saved.")
        
        print("\n🎉 SYNCHRONIZATION SESSION COMPLETE!")
        
    except KeyboardInterrupt:
        print("\n\n❌ Synchronization cancelled by user.")
    except Exception as e:
        print(f"\n❌ Error during synchronization: {e}")
        print("Please check your file paths and try again.")

if __name__ == "__main__":
    # Check if running interactively or as module
    import sys
    if len(sys.argv) > 1:
        # Command line usage
        print("Data Synchronization Tool")
        print("Usage:")
        print("  python sync.py  # Interactive mode")
        print("  sync_two_datasets('dataset1.csv', 'dataset2.csv', manual_mode=True)")
        print("  sync_two_datasets_time_shift('dataset1.csv', 'dataset2.csv', time_offset=55.0)")
        print("  auto_sync_two_datasets('dataset1.csv', 'dataset2.csv')  # Automatic detection")
    else:
        # Interactive mode
        main_interactive()
