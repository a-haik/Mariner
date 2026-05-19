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
            'Fatigue_Damage_Rate': normalized_fatigue_rate,
            'Mean_Power_Fluctuation_Intensity': df_chunk['POWER_TV_ENERGY'].abs().mean(),
        }
        return metrics
        
    def classify_phases(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Deterministic mapping using block-level kinematic analysis.
        
        'is_moving' is derived from the mean speed of the entire contiguous 
        status block to ensure stability in phase classification.
        """
        df = df.copy()
        raw_status = df['STATUS'].astype(str).str.lower().fillna('unknown')
        speed = df['SHIP SPEED(knots)'].fillna(0.0)

        # 1. Identify contiguous blocks based on raw status
        # This creates an ID that increments every time the status changes
        status_block_id = (raw_status != raw_status.shift()).cumsum()

        # 2. Calculate the mean speed for each block and map it to all rows in that block
        block_mean_speed = speed.groupby(status_block_id).transform('mean')

        # 3. Define is_moving based on the block's aggregate speed
        is_moving = block_mean_speed > self.speed_threshold

        # 4. Apply classification logic
        # Note: 'unknown' status is handled via fallback to kinematic data (is_moving)
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
        
        # default='Unknown' ensures that if no conditions are met, it is explicitly flagged
        df['PHASE'] = np.select(conditions, choices, default='Unknown')
        
        # 5. Retain trip logic for localized variance tracking (The Bricks)
        is_sea = df['PHASE'].str.startswith('Sea_')
        df['stay_id'] = (is_sea != is_sea.shift()).cumsum()
        
        return df

    def generate_block_registry(self, df: pd.DataFrame, source_file_name: str = "Unknown", 
                                merge_loitering: bool = True, unify_port_ops: bool = False) -> pd.DataFrame:
        if 'stay_id' not in df.columns:
            df = self.classify_phases(df)
            
        registry_entries = []
        unique_ids = df['stay_id'].unique()
        
        # Exclude partial boundary blocks (The first and last stay_ids are often truncated)
        valid_stay_ids = unique_ids[1:-1] if len(unique_ids) > 2 else []

        for s_id in valid_stay_ids:
            group = df[df['stay_id'] == s_id].copy()
            if group.empty:
                continue
                
            is_sea_block = group['PHASE'].iloc[0].startswith('Sea_')
            
            # --- 1. Sea Logic (Transit & Loitering) ---
            if is_sea_block:
                # Identify the dominant Transit phase for this block (Laden vs Ballast)
                transit_phases = group[group['PHASE'].str.contains('Transit')]['PHASE']
                base_phase = transit_phases.mode()[0] if not transit_phases.empty else 'Sea_Transit_Laden'
                
                # Normalization: Force all Transit phases to the base_phase
                # This ensures Transit1 -> Transit2 are merged even if Loitering is between them
                group.loc[group['PHASE'].str.contains('Transit'), 'PHASE'] = base_phase
                
                if merge_loitering:
                    # Force everything (including Loitering) to base_phase
                    group['PHASE'] = base_phase
                    
                # Group and compute
                for phase, sub in group.groupby('PHASE'):
                    metrics = self._compute_metrics(sub, source_file_name)
                    metrics.update({
                        'Stay_ID': s_id, 
                        'PHASE': phase, 
                        'Loitering_Handling': 'Merged' if merge_loitering else 'Separated'
                    })
                    registry_entries.append(metrics)

            # --- 2. Port Logic (Idle & Ops) ---
            else:
                if unify_port_ops:
                    # Identify the dominant operation (excluding Port_Idle)
                    ops = group[~group['PHASE'].isin(['Port_Idle'])]
                    main_op = ops['PHASE'].mode()[0] if not ops.empty else 'Port_Idle'
                    
                    # Force the whole block to the main operation
                    group['PHASE'] = main_op
                    
                    metrics = self._compute_metrics(group, source_file_name)
                    metrics.update({'Stay_ID': s_id, 'PHASE': main_op, 'Port_Handling': 'Unified'})
                    registry_entries.append(metrics)
                else:
                    # Keep everything distinct (Port_Idle, Port_Loading, etc.)
                    for phase, sub in group.groupby('PHASE'):
                        metrics = self._compute_metrics(sub, source_file_name)
                        metrics.update({'Stay_ID': s_id, 'PHASE': phase, 'Port_Handling': 'Separated'})
                        registry_entries.append(metrics)
                    
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