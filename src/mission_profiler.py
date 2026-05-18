import pandas as pd
import numpy as np
from typing import Dict, Callable

class MissionProfiler:
    """Classifies raw telemetry into physical mission phases using block-averaged kinematics."""
    
    def __init__(self, speed_threshold: float = 1.0, lhv_h2: float = 33.32, dt_min: float = 5.0):
        self.speed_threshold = speed_threshold
        self.lhv_h2 = lhv_h2
        self.dt_hours = dt_min / 60.0
        
        # Extensible degradation metrics dictionary mapping names to calculation functions.
        self.degradation_metrics: Dict[str, Callable[[pd.DataFrame], float]] = {
            'Intensive_L1_Power': lambda df: float(df['AE_POWER(kW)'].abs().mean()),
            'Intensive_L2_Volatility': lambda df: float(
                (df['POWER_TV_ENERGY'].sum() * (5 / 60.0)) / (len(df) * (5 / 60.0))
                if 'POWER_TV_ENERGY' in df.columns and len(df) > 0 else 0.0
            )
        }

    def classify_phases(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies physical and cargo tracking rules to determine raw states row-by-row."""
        df = df.copy()
        raw_status = df['STATUS'].str.lower()
        
        raw_block_id = (raw_status != raw_status.shift()).cumsum()
        block_mean_speed = df.groupby(raw_block_id)['SHIP SPEED(knots)'].transform('mean')
        
        is_sea = raw_status.isin(['laden', 'sea going', 'ballast']) | \
                 ((raw_status == 'idle') & (block_mean_speed > self.speed_threshold))
        
        stay_id = (is_sea != is_sea.shift()).cumsum()
        
        has_loading = df.groupby(stay_id)['STATUS'].transform(
            lambda x: x.str.lower().isin(['loading']).any()
        )
        has_discharge = df.groupby(stay_id)['STATUS'].transform(
            lambda x: x.str.lower().isin(['discharging', 'unloading']).any()
        )
        
        conditions = [
            is_sea & raw_status.isin(['laden', 'sea going']),
            is_sea & raw_status.isin(['ballast']),
            is_sea & (raw_status == 'idle'),  
            ~is_sea & has_loading,
            ~is_sea & has_discharge,
            ~is_sea 
        ]
        
        choices = [
            'Sea_Transit_Laden',
            'Sea_Transit_Ballast',
            'Sea_Loitering',
            'Port_Loading',
            'Port_Unloading',
            'Port_Idle'
        ]
        
        df['PHASE'] = np.select(conditions, choices, default='Unknown')
        
        phase_changes = df['PHASE'] != df['PHASE'].shift()
        df['PHASE_ID'] = phase_changes.cumsum()
        
        return df
    
    def add_degradation_metric(self, name: str, metric_func: Callable[[pd.DataFrame], float]) -> None:
        """Allows dynamic injection of new engineering or physical degradation metrics."""
        self.degradation_metrics[name] = metric_func

    def _compute_metrics(self, df_chunk: pd.DataFrame, source_file: str) -> dict:
        """Helper to calculate mathematical invariants over a single isolated slice."""
        duration = len(df_chunk) * self.dt_hours
        energy_kwh = df_chunk['AE_POWER(kW)'].sum() * self.dt_hours
        mean_power = df_chunk['AE_POWER(kW)'].mean()
        
        metrics = {
            'Source_File': source_file,
            'Start_Time': df_chunk.index.min().strftime('%Y-%m-%d %H:%M'),
            'Duration_h': duration,
            'Energy_kWh': energy_kwh,
            'Mean_Power_kW': mean_power
        }
        
        metrics['H2_Rate_Lower_kg_h'] = mean_power / (0.55 * self.lhv_h2)
        metrics['H2_Rate_Upper_kg_h'] = mean_power / (0.45 * self.lhv_h2)
        
        metrics['H2_Cons_Lower_kg'] = energy_kwh / (0.55 * self.lhv_h2)
        metrics['H2_Cons_Upper_kg'] = energy_kwh / (0.45 * self.lhv_h2)
        
        for name, func in self.degradation_metrics.items():
            metrics[name] = func(df_chunk)
            
        return metrics

    def generate_phase_registry(self, df: pd.DataFrame, source_file_name: str = "Unknown") -> pd.DataFrame:
        """
        Processes telemetry to yield a relational registry of unique trips,
        excluding partial boundary blocks and generating explicit loitering entries.
        """
        df_labeled = self.classify_phases(df)
        raw_status = df_labeled['STATUS'].str.lower()
        
        raw_block_id = (raw_status != raw_status.shift()).cumsum()
        block_mean_speed = df_labeled.groupby(raw_block_id)['SHIP SPEED(knots)'].transform('mean')
        
        is_sea = raw_status.isin(['laden', 'sea going', 'ballast']) | \
                 ((raw_status == 'idle') & (block_mean_speed > self.speed_threshold))
        
        stay_id = (is_sea != is_sea.shift()).cumsum()
        
        # Identify the censored boundary blocks
        min_stay_id = stay_id.min()
        max_stay_id = stay_id.max()
        
        registry_entries = []
        
        for s_id, group in df_labeled.groupby(stay_id):
            # EXCLUDE partial blocks at the start and end of the dataset
            if s_id == min_stay_id or s_id == max_stay_id:
                continue
                
            is_sea_block = group['PHASE'].iloc[0].startswith('Sea_')
            
            if not is_sea_block:
                phase_label = group['PHASE'].iloc[0]
                metrics = self._compute_metrics(group, source_file_name)
                metrics.update({'Stay_ID': s_id, 'PHASE': phase_label, 'Loitering_Handling': 'Included'})
                registry_entries.append(metrics)
            
            else:
                is_laden = (group['PHASE'] == 'Sea_Transit_Laden').sum() >= (group['PHASE'] == 'Sea_Transit_Ballast').sum()
                base_phase = 'Sea_Transit_Laden' if is_laden else 'Sea_Transit_Ballast'
                
                metrics_with = self._compute_metrics(group, source_file_name)
                metrics_with.update({'Stay_ID': s_id, 'PHASE': base_phase, 'Loitering_Handling': 'With_Loitering'})
                registry_entries.append(metrics_with)
                
                group_without = group[group['PHASE'] != 'Sea_Loitering']
                
                if not group_without.empty:
                    metrics_without = self._compute_metrics(group_without, source_file_name)
                    metrics_without.update({'Stay_ID': s_id, 'PHASE': base_phase, 'Loitering_Handling': 'Without_Loitering'})
                    registry_entries.append(metrics_without)
                    
        return pd.DataFrame(registry_entries)