# src/mission_profiler.py
import pandas as pd
import numpy as np
import rainflow
from typing import Dict
from src.config import PHYSICS

class MissionProfiler:
    """Handles deterministic modes classification and profile blocks generation."""
    
    def __init__(self, speed_threshold: float = PHYSICS.SPEED_THRESHOLD_KNOTS, dt_min: float = 5.0):
        self.speed_threshold = speed_threshold
        self.dt_hours = dt_min / 60.0

    def _compute_normalized_rainflow(self, power_series: pd.Series, duration_h: float) -> float:
        """
        Calculates the time-normalized, module-scaled Expected Fatigue Rate.
        """
        if power_series.isna().all() or len(power_series) < 2:
            return 0.0
            
        # Extract raw cycles (amplitude, mean, count, start, end)
        cycles = rainflow.extract_cycles(power_series.dropna().values)
        
        total_normalized_damage = 0.0
        for amplitude, _, count, _, _ in cycles:
            # Scale the raw amplitude (kW) to an intensive module unit
            scaled_amplitude = amplitude / PHYSICS.P_BASE_MODULE_KW
            
            # Palmgren-Miner damage accumulation
            total_normalized_damage += count * (scaled_amplitude ** PHYSICS.FATIGUE_EXPONENT_K)
            
        # Divide by duration to yield an intensive rate per hour
        fatigue_rate = total_normalized_damage / (duration_h + PHYSICS.EPSILON)
        return fatigue_rate

    def _compute_metrics(self, df_chunk: pd.DataFrame, source_file: str) -> dict:
        duration = len(df_chunk) * self.dt_hours
        mean_power = df_chunk['AE_POWER(kW)'].mean()
        
        metrics = {
            'Source_File': source_file,
            'Start_Time': df_chunk.index.min().strftime('%Y-%m-%d %H:%M'),
            'Duration_h': duration,
            'Energy_kWh': mean_power * duration,
            'Mean_Power_kW': mean_power,
            'H2_Rate_Lower_kg_h': mean_power / (PHYSICS.ETA_UPPER * PHYSICS.LHV_H2_KWH_KG),
            'H2_Rate_Upper_kg_h': mean_power / (PHYSICS.ETA_LOWER * PHYSICS.LHV_H2_KWH_KG),
            'Relative_Fatigue_Activity_Rate': self._compute_normalized_rainflow(df_chunk['AE_POWER(kW)'], duration),
            'Mean_Power_Fluctuation_Intensity': df_chunk['POWER_TV_ENERGY'].abs().mean(),
        }
        return metrics
        
    def classify_modes(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        raw_status = df['STATUS'].astype(str).str.lower().fillna('unknown')
        speed = df['SHIP SPEED(knots)'].fillna(0.0)

        status_block_id = (raw_status != raw_status.shift()).cumsum()
        block_mean_speed = speed.groupby(status_block_id).transform('mean')
        is_moving = block_mean_speed > self.speed_threshold

        conditions = [
            raw_status.isin(['laden', 'sea going']) | ((raw_status == 'unknown') & is_moving),
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
        df['MODE'] = np.select(conditions, choices, default='Unknown')
        is_sea = df['MODE'].str.startswith('Sea_')
        df['stay_id'] = (is_sea != is_sea.shift()).cumsum()
        return df

    def generate_block_registry(self, df: pd.DataFrame, source_file_name: str = "Unknown", 
                                merge_loitering: bool = True, unify_port_ops: bool = True) -> pd.DataFrame:
        if 'stay_id' not in df.columns:
            df = self.classify_modes(df)
            
        registry_entries = []
        unique_ids = df['stay_id'].unique()
        valid_stay_ids = unique_ids[1:-1] if len(unique_ids) > 2 else []

        for s_id in valid_stay_ids:
            group = df[df['stay_id'] == s_id].copy()
            if group.empty:
                continue
                
            is_sea_block = group['MODE'].iloc[0].startswith('Sea_')
            
            if is_sea_block:
                transit_modes = group[group['MODE'].str.contains('Transit')]['MODE']
                base_mode = transit_modes.mode()[0] if not transit_modes.empty else 'Sea_Transit_Laden'
                group.loc[group['MODE'].str.contains('Transit'), 'MODE'] = base_mode
                if merge_loitering:
                    group.loc[group['MODE'] == 'Sea_Loitering', 'MODE'] = base_mode
                
                for mode, sub in group.groupby('MODE'):
                    metrics = self._compute_metrics(sub, source_file_name)
                    metrics.update({'Stay_ID': s_id, 'MODE': mode, 'Loitering_Handling': 'Merged' if merge_loitering else 'Separated'})
                    registry_entries.append(metrics)
            else:
                if unify_port_ops:
                    ops = group[~group['MODE'].isin(['Port_Idle'])]
                    main_op = ops['MODE'].mode()[0] if not ops.empty else 'Port_Idle'
                    group['MODE'] = main_op
                    metrics = self._compute_metrics(group, source_file_name)
                    metrics.update({'Stay_ID': s_id, 'MODE': main_op, 'Port_Handling': 'Unified'})
                    registry_entries.append(metrics)
                else:
                    for mode, sub in group.groupby('MODE'):
                        metrics = self._compute_metrics(sub, source_file_name)
                        metrics.update({'Stay_ID': s_id, 'MODE': mode, 'Port_Handling': 'Separated'})
                        registry_entries.append(metrics)
                    
        return pd.DataFrame(registry_entries)

    def extract_global_statistics(self, registry_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregates global stats directly from the block registry using duration-weighted means.
        This guarantees mathematical consistency (Conservation of Energy) between the bricks and global metrics.
        """
        if registry_df.empty:
            return pd.DataFrame()
        
        total_registry_duration = registry_df['Duration_h'].sum()
            
        stats = []
        for mode, group in registry_df.groupby('MODE'):
            total_hours = group['Duration_h'].sum()
            if total_hours == 0:
                continue
                
            def weighted_mean(col):
                return (group[col] * group['Duration_h']).sum() / total_hours

            stats.append({
                'MODE': mode,
                'Mean_Power_kW': weighted_mean('Mean_Power_kW'),
                'Mean_H2_Rate_Lower_kg_h': weighted_mean('H2_Rate_Lower_kg_h'),
                'Mean_H2_Rate_Upper_kg_h': weighted_mean('H2_Rate_Upper_kg_h'),
                'Mean_Power_Fluctuation_Intensity': weighted_mean('Mean_Power_Fluctuation_Intensity'),
                'Relative_Fatigue_Activity_Rate': weighted_mean('Relative_Fatigue_Activity_Rate'),
                'Number_of_Blocks': len(group),
                'Total_Logged_Hours': total_hours,
                'Time_Fraction': total_hours / total_registry_duration if total_registry_duration > 0 else 0
            })
            
        return pd.DataFrame(stats).set_index('MODE')


class ScenarioEvaluator:
    """Linear Engine for continuous 1000h scenarios."""
    def __init__(self, global_stats: pd.DataFrame):
        self.stats = global_stats

    def _print_report(self, results: dict):
        """Helper for a clean console summary."""
        print("\n" + "="*45)
        print("SCENARIO EVALUATION REPORT")
        print("="*45)
        print(f"Total Duration    : {results['Scenario_Hours']:.1f} h")
        print(f"Energy Demanded   : {results['Expected_Energy_kWh']:.1f} kWh")
        print("-" * 45)
        print(f"H2 Consumption    : [{results['Expected_Total_H2_Lower_kg']:.2f} - "
              f"{results['Expected_Total_H2_Upper_kg']:.2f}] kg")
        print(f"Fatigue Index     : {results['Expected_Total_Fatigue']:.4f} units")
        print(f"Fluctuation Index : {results['Expected_Total_Fluctuation']:.2f} kW²/s²")
        print("="*45 + "\n")

    def evaluate(self, time_weights_hours: Dict[str, float], print_results: bool = False) -> dict:
        w = pd.Series(time_weights_hours).reindex(self.stats.index).fillna(0.0)
        
        expected_h2_lower = np.dot(w, self.stats['Mean_H2_Rate_Lower_kg_h'])
        expected_h2_upper = np.dot(w, self.stats['Mean_H2_Rate_Upper_kg_h'])
        expected_energy = np.dot(w, self.stats['Mean_Power_kW'])
        
        # Total fatigue over scenario = sum(Rate_i * Time_i)
        expected_total_fatigue = np.dot(w, self.stats['Relative_Fatigue_Activity_Rate'])
        expected_total_fluctuation = np.dot(w, self.stats['Mean_Power_Fluctuation_Intensity'])
        
        results = {
            'Scenario_Hours': w.sum(),
            'Expected_Energy_kWh': expected_energy,
            'Expected_Total_H2_Lower_kg': expected_h2_lower,
            'Expected_Total_H2_Upper_kg': expected_h2_upper,
            'Expected_Total_Fatigue': expected_total_fatigue,
            'Expected_Total_Fluctuation': expected_total_fluctuation,
            'Weights_Applied': w.to_dict()
        }
        
        if print_results:
            self._print_report(results)
            
        return results