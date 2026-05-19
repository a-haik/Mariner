# src/mission_profiler.py
import pandas as pd
import numpy as np
import rainflow
from typing import Dict
from src.config import PHYSICS

class MissionProfiler:
    """Handles deterministic phase classification and brick generation."""
    
    def __init__(self, speed_threshold: float = PHYSICS.SPEED_THRESHOLD_KNOTS, dt_min: float = 5.0):
        self.speed_threshold = speed_threshold
        self.dt_hours = dt_min / 60.0

        
    def classify_phases(self, df: pd.DataFrame) -> pd.DataFrame:
        """Deterministic mapping with Kinematic Fallback."""
        df = df.copy()
        raw_status = df['STATUS'].astype(str).str.lower().fillna('unknown')
        speed = df['SHIP SPEED(knots)'].fillna(0.0)

        is_moving = speed > self.speed_threshold

        conditions = [
            raw_status.isin(['laden', 'sea going']) | ((raw_status == 'unknown') & (speed > self.speed_threshold)),
            raw_status.isin(['ballast']),
            (raw_status == 'idle') & is_moving,
            ~is_moving & raw_status.isin(['loading']),
            ~is_moving & raw_status.isin(['discharging', 'unloading']),
            ~is_moving & raw_status.isin(['idle', 'port_idle', 'unknown']) 
        ]
        
        choices = [
            'Sea_Transit_Laden', 'Sea_Transit_Ballast', 'Sea_Loitering',
            'Port_Loading', 'Port_Unloading', 'Port_Idle'
        ]
        
        df['PHASE'] = np.select(conditions, choices, default='Unknown')
        
        # Retain trip logic for localized variance tracking (The Bricks)
        is_sea = df['PHASE'].str.startswith('Sea_')
        df['stay_id'] = (is_sea != is_sea.shift()).cumsum()
        
        return df
    
    def _compute_normalized_rainflow(self, power_series: pd.Series, duration_h: float, k: float = 2.0) -> float:
        """
        Calculates the intensive Expected Fatigue Rate using Rainflow counting.
        k is the fatigue exponent (assumed 2.0 for variance-like scaling).
        """
        if power_series.isna().all() or len(power_series) < 2:
            return 0.0
            
        # rainflow.extract_cycles returns (range, mean, count, i_start, i_end)
        # count is usually 1.0 or 0.5 (for half cycles)
        cycles = rainflow.extract_cycles(power_series.dropna().values)
        
        # Calculate Palmgren-Miner damage: sum(count * amplitude^k)
        total_damage = sum(count * (amplitude ** k) for amplitude, _, count, _, _ in cycles)
        
        # Normalize by duration to get the intensive rate
        fatigue_rate = total_damage / (duration_h + PHYSICS.EPSILON)
        return fatigue_rate

    def _compute_metrics(self, df_chunk: pd.DataFrame, source_file: str) -> dict:
        """Helper to calculate mathematical invariants over a single isolated slice."""
        duration = len(df_chunk) * self.dt_hours
        mean_power = df_chunk['AE_POWER(kW)'].mean()
        energy_kwh = mean_power * duration
        
        # Calculate our new intensive fatigue metric
        normalized_fatigue_rate = self._compute_normalized_rainflow(df_chunk['AE_POWER(kW)'], duration)
        
        metrics = {
            'Source_File': source_file,
            'Start_Time': df_chunk.index.min().strftime('%Y-%m-%d %H:%M'),
            'Duration_h': duration,
            'Energy_kWh': energy_kwh,
            'Mean_Power_kW': mean_power,
            'H2_Rate_Lower_kg_h': mean_power / (PHYSICS.ETA_UPPER * PHYSICS.LHV_H2_KWH_KG),
            'H2_Rate_Upper_kg_h': mean_power / (PHYSICS.ETA_LOWER * PHYSICS.LHV_H2_KWH_KG),
            'Fatigue_Damage_Rate ': normalized_fatigue_rate,
            'Mean_Power_Fluctuation_Intensity': df_chunk['POWER_TV_ENERGY'].abs().mean(),
        }
        return metrics

    def generate_phase_registry(self, df: pd.DataFrame, source_file_name: str = "Unknown") -> pd.DataFrame:
        """
        Localized Aggregation for Brick Visualization.
        Reconstructs the discrete blocks required for plot_brick_space.
        """
        if 'stay_id' not in df.columns:
            df = self.classify_phases(df)
            
        registry_entries = []
        stay_ids = df['stay_id'].unique()
        
        # Exclude partial boundary blocks at the start and end
        valid_stay_ids = stay_ids[1:-1] if len(stay_ids) > 2 else stay_ids

        for s_id in valid_stay_ids:
            group = df[df['stay_id'] == s_id]
            is_sea_block = group['PHASE'].iloc[0].startswith('Sea_')
            
            if not is_sea_block:
                for phase_label, sub_group in group.groupby('PHASE'):
                    metrics = self._compute_metrics(sub_group, source_file_name)
                    metrics.update({
                        'Stay_ID': s_id, 
                        'PHASE': phase_label, 
                    })
                    registry_entries.append(metrics)
            else:
                # Determine primary transit mode for this sea block
                is_laden = (group['PHASE'] == 'Sea_Transit_Laden').sum() >= (group['PHASE'] == 'Sea_Transit_Ballast').sum()
                base_phase = 'Sea_Transit_Laden' if is_laden else 'Sea_Transit_Ballast'
                
                # Metric 1: The true physical trip (With Loitering)
                metrics_with = self._compute_metrics(group, source_file_name)
                metrics_with.update({'Stay_ID': s_id, 'PHASE': base_phase, 'With_Loitering': True})
                registry_entries.append(metrics_with)
            
                # Metric 2: The compressed synthetic trip (Without Loitering)
                group_without = group[group['PHASE'] != 'Sea_Loitering']
                if not group_without.empty:
                    metrics_without = self._compute_metrics(group_without, source_file_name)
                    metrics_without.update({'Stay_ID': s_id, 'PHASE': base_phase, 'Loitering_Handling': 'Without_Loitering'})
                    registry_entries.append(metrics_without)
                    
        return pd.DataFrame(registry_entries)

    def extract_global_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Collapses time-series into intensive state vectors for the ScenarioEvaluator."""
        if 'H2_Rate_Lower_kg_h' not in df.columns:
             df['H2_Rate_Lower_kg_h'] = df['AE_POWER(kW)'] / (PHYSICS.ETA_UPPER * PHYSICS.LHV_H2_KWH_KG)
        if 'H2_Rate_Upper_kg_h' not in df.columns:
             df['H2_Rate_Upper_kg_h'] = df['AE_POWER(kW)'] / (PHYSICS.ETA_LOWER * PHYSICS.LHV_H2_KWH_KG)

        valid_df = df[df['PHASE'] != 'Unknown']

        stats = valid_df.groupby('PHASE').agg(
            Mean_Power_kW=('AE_POWER(kW)', 'mean'),
            Std_Power_kW=('AE_POWER(kW)', 'std'),
            Mean_H2_Rate_Lower_kg_h=('H2_Rate_Lower_kg_h', 'mean'),
            Mean_H2_Rate_Upper_kg_h=('H2_Rate_Upper_kg_h', 'mean'),
            Mean_Power_Fluctuation_Intensity=('POWER_TV_ENERGY', 'mean'),
            Sample_Count=('AE_POWER(kW)', 'count')
        ).fillna(0)


        def calculate_phase_fatigue(group):
            # Calculate duration of this specific group in hours
            duration = len(group) * self.dt_hours
            return self._compute_normalized_rainflow(group['AE_POWER(kW)'], duration)

        fatigue_rates = valid_df.groupby('PHASE').apply(calculate_phase_fatigue, include_groups=False)
        stats['Fatigue_Damage_Rate'] = fatigue_rates
        stats['Total_Logged_Hours'] = stats['Sample_Count'] * self.dt_hours
        
        return stats.fillna(0)


class ScenarioEvaluator:
    """Linear Engine for continuous 1000h scenarios."""
    def __init__(self, global_stats: pd.DataFrame):
        self.stats = global_stats

    def evaluate(self, time_weights_hours: Dict[str, float]) -> dict:
        w = pd.Series(time_weights_hours).reindex(self.stats.index).fillna(0.0)
        
        expected_h2_lower = np.dot(w, self.stats['Mean_H2_Rate_Lower_kg_h'])
        expected_h2_upper = np.dot(w, self.stats['Mean_H2_Rate_Upper_kg_h'])
        expected_energy = np.dot(w, self.stats['Mean_Power_kW'])
        expected_total_fatigue = np.dot(w, self.stats['Fatigue_Damage_Rate'])
        expected_total_fluctuation = np.dot(w, self.stats['Mean_Power_Fluctuation_Intensity'])
        
        # Linearly combine the intensive fatigue rates
        # Total fatigue over scenario = sum(Rate_i * Time_i)
        expected_total_fatigue = np.dot(w, self.stats['Fatigue_Damage_Rate'])
        
        return {
            'Scenario_Hours': w.sum(),
            'Expected_Energy_kWh': expected_energy,
            'Expected_Total_H2_Lower_kg': expected_h2_lower,
            'Expected_Total_H2_Upper_kg': expected_h2_upper,
            'Expected_Total_Fatigue': expected_total_fatigue,
            'Expected_Total_Fluctuation': expected_total_fluctuation,
            'Weights_Applied': w.to_dict()
        }