import pandas as pd
import numpy as np
from typing import Dict

class MissionProfiler:
    """Handles deterministic phase classification, brick generation, and global statistical aggregation."""
    
    def __init__(self, speed_threshold: float = 1.0, lhv_h2: float = 33.32, dt_min: float = 5.0):
        self.speed_threshold = speed_threshold
        self.lhv_h2 = lhv_h2
        self.dt_hours = dt_min / 60.0
        self.eta_upper = 0.55 # Baseline / Best Case
        self.eta_lower = 0.45 # Degraded / Worst Case
        
    def classify_phases(self, df: pd.DataFrame) -> pd.DataFrame:
        """Phase 1: Deterministic mapping with Kinematic Fallback."""
        df = df.copy()
        raw_status = df['STATUS'].astype(str).str.lower().fillna('unknown')
        speed = df['SHIP SPEED(knots)'].fillna(0.0)

        is_moving = speed > self.speed_threshold

        conditions = [
            raw_status.isin(['laden', 'sea going']) | ((raw_status == 'unknown') & (speed > 5.0)),
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

    def _compute_metrics(self, df_chunk: pd.DataFrame, source_file: str) -> dict:
        """Helper to calculate mathematical invariants over a single isolated slice."""
        duration = len(df_chunk) * self.dt_hours
        mean_power = df_chunk['AE_POWER(kW)'].mean()
        energy_kwh = mean_power * duration
        
        l2_volatility = df_chunk['POWER_TV_ENERGY'].mean() if 'POWER_TV_ENERGY' in df_chunk.columns else 0.0
        
        metrics = {
            'Source_File': source_file,
            'Start_Time': df_chunk.index.min().strftime('%Y-%m-%d %H:%M'),
            'Duration_h': duration,
            'Energy_kWh': energy_kwh,
            'Mean_Power_kW': mean_power,
            'H2_Rate_Lower_kg_h': mean_power / (self.eta_upper * self.lhv_h2),
            'H2_Rate_Upper_kg_h': mean_power / (self.eta_lower * self.lhv_h2),
            'H2_Cons_Lower_kg': energy_kwh / (self.eta_upper * self.lhv_h2),
            'H2_Cons_Upper_kg': energy_kwh / (self.eta_lower * self.lhv_h2),
            'Intensive_L2_Volatility': l2_volatility,
            'Intensive_L1_Power': df_chunk['AE_POWER(kW)'].abs().mean()
        }
        return metrics

    def generate_phase_registry(self, df: pd.DataFrame, source_file_name: str = "Unknown") -> pd.DataFrame:
        """
        Phase 2a: Localized Aggregation for Brick Visualization.
        Reconstructs the discrete blocks required for plot_brick_space.
        """
        if 'stay_id' not in df.columns:
            df = self.classify_phases(df)
            
        registry_entries = []
        stay_ids = df['stay_id'].unique()
        
        # Exclude partial boundary blocks at the start and end of the dataset
        valid_stay_ids = stay_ids[1:-1] if len(stay_ids) > 2 else stay_ids

        for s_id in valid_stay_ids:
            group = df[df['stay_id'] == s_id]
            is_sea_block = group['PHASE'].iloc[0].startswith('Sea_')
            
            if not is_sea_block:
                phase_label = group['PHASE'].iloc[0]
                metrics = self._compute_metrics(group, source_file_name)
                metrics.update({'Stay_ID': s_id, 'PHASE': phase_label, 'Loitering_Handling': 'Included'})
                registry_entries.append(metrics)
            else:
                # Determine primary transit mode for this sea block
                is_laden = (group['PHASE'] == 'Sea_Transit_Laden').sum() >= (group['PHASE'] == 'Sea_Transit_Ballast').sum()
                base_phase = 'Sea_Transit_Laden' if is_laden else 'Sea_Transit_Ballast'
                
                # Metric 1: The true physical trip (With Loitering)
                metrics_with = self._compute_metrics(group, source_file_name)
                metrics_with.update({'Stay_ID': s_id, 'PHASE': base_phase, 'Loitering_Handling': 'With_Loitering'})
                registry_entries.append(metrics_with)
                
                # Metric 2: The compressed synthetic trip (Without Loitering)
                group_without = group[group['PHASE'] != 'Sea_Loitering']
                if not group_without.empty:
                    metrics_without = self._compute_metrics(group_without, source_file_name)
                    metrics_without.update({'Stay_ID': s_id, 'PHASE': base_phase, 'Loitering_Handling': 'Without_Loitering'})
                    registry_entries.append(metrics_without)
                    
        return pd.DataFrame(registry_entries)

    def extract_global_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Phase 2b: Collapses time-series into intensive state vectors for the ScenarioEvaluator."""
        if 'H2_Rate_Lower_kg_h' not in df.columns:
             df['H2_Rate_Lower_kg_h'] = df['AE_POWER(kW)'] / (self.eta_upper * self.lhv_h2)
        if 'H2_Rate_Upper_kg_h' not in df.columns:
             df['H2_Rate_Upper_kg_h'] = df['AE_POWER(kW)'] / (self.eta_lower * self.lhv_h2)

        valid_df = df[df['PHASE'] != 'Unknown']

        stats = valid_df.groupby('PHASE').agg(
            Mean_Power_kW=('AE_POWER(kW)', 'mean'),
            Std_Power_kW=('AE_POWER(kW)', 'std'),
            Mean_H2_Rate_Lower_kg_h=('H2_Rate_Lower_kg_h', 'mean'),
            Mean_H2_Rate_Upper_kg_h=('H2_Rate_Upper_kg_h', 'mean'),
            Mean_L2_Volatility=('POWER_TV_ENERGY', 'mean'),
            Sample_Count=('AE_POWER(kW)', 'count')
        ).fillna(0)

        stats['Total_Logged_Hours'] = stats['Sample_Count'] * self.dt_hours
        return stats


class ScenarioEvaluator:
    """Phase 3: Linear Engine for continuous 1000h scenarios."""
    def __init__(self, global_stats: pd.DataFrame):
        self.stats = global_stats

    def evaluate(self, time_weights_hours: Dict[str, float]) -> dict:
        w = pd.Series(time_weights_hours).reindex(self.stats.index).fillna(0.0)
        
        expected_h2_lower = np.dot(w, self.stats['Mean_H2_Rate_Lower_kg_h'])
        expected_h2_upper = np.dot(w, self.stats['Mean_H2_Rate_Upper_kg_h'])
        expected_l2 = np.dot(w, self.stats['Mean_L2_Volatility'])
        expected_energy = np.dot(w, self.stats['Mean_Power_kW'])
        
        return {
            'Scenario_Hours': w.sum(),
            'Expected_Energy_kWh': expected_energy,
            'Expected_Total_H2_Lower_kg': expected_h2_lower,
            'Expected_Total_H2_Upper_kg': expected_h2_upper,
            'Expected_L2_Fatigue_Index': expected_l2,
            'Weights_Applied': w.to_dict()
        }