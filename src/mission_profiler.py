# src/mission_profiler.py
import pandas as pd
import numpy as np
import rainflow
from typing import Dict, Any
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


class ScenarioManager:
    """
    Unified evaluation engine for MARINER 1000-hour testing scenarios.
    Handles data-driven weight generation, low-cost statistical extraction,
    and runs the master execution pipeline to compile results for reporting.
    """
    
    def __init__(self, registry_df: pd.DataFrame, global_stats: pd.DataFrame):
        self.registry = registry_df
        self.stats = global_stats
        self.total_hours = registry_df['Duration_h'].sum() if not registry_df.empty else 0.0
        
        # Derive empirical baseline fractions from the master registry
        self.base_fractions = {}
        if not registry_df.empty:
            for mode, group in registry_df.groupby('MODE'):
                self.base_fractions[mode] = group['Duration_h'].sum() / self.total_hours

    def _extract_low_cost_stats(self) -> pd.DataFrame:
        """Filters the baseline registry to compute stats from the 25% lowest H2 cases per mode."""
        low_cost_blocks = []
        for mode, group in self.registry.groupby('MODE'):
            # Select the bottom 25% of cases based on lower-bound H2 flow rate
            n_select = max(1, int(len(group) * 0.25))
            sorted_group = group.sort_values(by='H2_Rate_Lower_kg_h')
            low_cost_blocks.append(sorted_group.head(n_select))
            
        low_cost_registry = pd.concat(low_cost_blocks, ignore_index=True)
        
        # Aggregate duration-weighted statistics for the low-cost registry subset
        stats_list = []
        for mode, group in low_cost_registry.groupby('MODE'):
            t_hours = group['Duration_h'].sum()
            def weighted_mean(col):
                return (group[col] * group['Duration_h']).sum() / t_hours
            stats_list.append({
                'MODE': mode,
                'Mean_Power_kW': weighted_mean('Mean_Power_kW'),
                'Mean_H2_Rate_Lower_kg_h': weighted_mean('H2_Rate_Lower_kg_h'),
                'Mean_H2_Rate_Upper_kg_h': weighted_mean('H2_Rate_Upper_kg_h'),
                'Relative_Fatigue_Activity_Rate': weighted_mean('Relative_Fatigue_Activity_Rate'),
                'Mean_Power_Fluctuation_Intensity': weighted_mean('Mean_Power_Fluctuation_Intensity')
            })
        return pd.DataFrame(stats_list).set_index('MODE')

    def _generate_scenarios_weights(self, total_target_h: float) -> Dict[str, Dict[str, float]]:
        """Generates target hourly distributions for the defined MARINER profiles."""
        scenarios = {}
        port_heavy_modes = ['Port_Loading', 'Port_Unloading']

        # 1. Baseline
        scenarios['Baseline'] = {m: f * total_target_h for m, f in self.base_fractions.items()}

        # 2. Shore Power: Replace loading/unloading with Port Idle hotel loads
        sp_fracs = self.base_fractions.copy()
        heavy_time = sum(sp_fracs.pop(m, 0.0) for m in port_heavy_modes)
        sp_fracs['Port_Idle'] = sp_fracs.get('Port_Idle', 0.0) + heavy_time
        scenarios['Shore_Power'] = {m: f * total_target_h for m, f in sp_fracs.items()}

        # 3. PEMFC Off Port: Drop port operations without replacement, re-normalize remaining times
        fc_off = {m: f for m, f in self.base_fractions.items() if m not in port_heavy_modes}
        norm = sum(fc_off.values())
        scenarios['PEMFC_Off_Port'] = {m: (f / norm) * total_target_h for m, f in fc_off.items()}

        # 4. Data-Driven Long Trips: Derived from empirical transit length distributions
        transit_blocks = self.registry[self.registry['MODE'].str.contains('Transit', na=False)]
        if transit_blocks.empty:
            scenarios['Long_Trips'] = scenarios['Baseline']
        else:
            median_transit_length = transit_blocks['Duration_h'].median()
            lt_fracs = {}
            for mode, group in self.registry.groupby('MODE'):
                if 'Transit' in mode:
                    long_contribution = group[group['Duration_h'] >= median_transit_length]['Duration_h'].sum()
                    lt_fracs[mode] = max(long_contribution, PHYSICS.EPSILON)
                else:
                    lt_fracs[mode] = group['Duration_h'].sum() * 0.5  # Reduce port transit frequencies
            
            norm_lt = sum(lt_fracs.values())
            scenarios['Long_Trips'] = {m: (f / norm_lt) * total_target_h for m, f in lt_fracs.items()}

        return scenarios

    def _evaluate_single_matrix(self, weights: Dict[str, float], stats_matrix: pd.DataFrame) -> Dict[str, float]:
        """Calculates the linear combination of weighted operational metrics."""
        w_series = pd.Series(weights).reindex(stats_matrix.index).fillna(0.0)
        return {
            'Expected_Energy_kWh': float(np.dot(w_series, stats_matrix['Mean_Power_kW'])),
            'H2_Consumed_Lower_kg': float(np.dot(w_series, stats_matrix['Mean_H2_Rate_Lower_kg_h'])),
            'H2_Consumed_Upper_kg': float(np.dot(w_series, stats_matrix['Mean_H2_Rate_Upper_kg_h'])),
            'Accumulated_Fatigue': float(np.dot(w_series, stats_matrix['Relative_Fatigue_Activity_Rate'])),
            'Accumulated_Fluctuation': float(np.dot(w_series, stats_matrix['Mean_Power_Fluctuation_Intensity']))
        }
    
    def _generate_scenario_summary_table(self, compiled_results: dict) -> pd.DataFrame:
        """
        Transforms the compiled scenarios output dictionary into a clean, 
        structured pandas DataFrame for engineering report formatting.
        """
        records = []
        
        for scenario_name, variants in compiled_results.items():
            for variant_type in ['Standard', 'LowCost']:
                metrics = variants[variant_type]
                records.append({
                    'Scenario Profile': scenario_name,
                    'Data Strategy': 'Full Run Average' if variant_type == 'Standard' else 'Bottom 25% Optimized',
                    'Energy Demanded (kWh)': round(metrics['Expected_Energy_kWh'], 1),
                    'H2 Mass Min [η=0.55] (kg)': round(metrics['H2_Consumed_Lower_kg'], 1),
                    'H2 Mass Max [η=0.45] (kg)': round(metrics['H2_Consumed_Upper_kg'], 1),
                    'Fatigue Index': round(metrics['Accumulated_Fatigue'], 5),
                    'Power Fluctuation Intensity': round(metrics['Accumulated_Fluctuation'], 2)
                })
            
        df_summary = pd.DataFrame(records)
        return df_summary

    def run_full_scenario_pipeline(self, total_target_h: float = 1000.0, print_summary: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        Master execution pipeline. Generates weights, runs evaluations for both 
        Standard and Low-Cost data distributions, and compiles the nested output.
        """
        # 1. Compute low-cost statistics matrix up front
        low_cost_stats = self._extract_low_cost_stats()
        
        # 2. Generate weights for all targets
        scenarios_weights = self._generate_scenarios_weights(total_target_h)
        
        compiled_results = {}
        
        # 3. Loop through configurations and evaluate both statistical layers
        for scenario_name, weights in scenarios_weights.items():
            compiled_results[scenario_name] = {
                'Standard': self._evaluate_single_matrix(weights, self.stats),
                'LowCost': self._evaluate_single_matrix(weights, low_cost_stats),
                'Time_Weights_Hours': weights
            }

        if print_summary:
            print(self._generate_scenario_summary_table(compiled_results))     
        return compiled_results