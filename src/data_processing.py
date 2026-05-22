# src/data_processing.py
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter, butter, filtfilt
from typing import Literal, List
from .config import PHYSICS, FILTERS

def calc_trig_headings(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates trigonometric encodings for vessel heading."""
    if 'HEADING(degree)' in df.columns:
        rads = np.radians(df['HEADING(degree)'])
        return df.assign(
            HEADING_SIN=np.sin(rads),
            HEADING_COS=np.cos(rads)
        )
    return df

def _compute_inline_battery_specs(df: pd.DataFrame, filtered_power: pd.Series, ignore_modes: list = None) -> dict:
    """
    Deduces battery properties (capacity required) by isolating continuous voyage blocks 
    and integrating the difference between raw ship demand and the filtered FC output.
    """
    raw_power = df['AE_POWER(kW)']
    p_batt = raw_power - filtered_power
    
    if ignore_modes and 'MODE' in df.columns:
        ignored_clean = [str(m).lower() for m in ignore_modes]
        status_mask = df['MODE'].astype(str).str.lower().isin(ignored_clean)
        p_batt = p_batt.mask(status_mask, 0.0)
        
    dt_hours = raw_power.index.to_series().diff().median().total_seconds() / 3600.0

    block_ids = df['stay_id'] if 'stay_id' in df.columns else pd.Series(1, index=df.index)
    max_excursion = 0.0
    
    for _, group_idx in p_batt.groupby(block_ids).groups.items():
        p_batt_block = p_batt.loc[group_idx]
        if len(p_batt_block) < 2:
            continue
            
        p_batt_centered = p_batt_block - p_batt_block.mean()
        e_cumulative = (p_batt_centered * dt_hours).cumsum()
        
        block_excursion = e_cumulative.max() - e_cumulative.min()
        max_excursion = max(max_excursion, block_excursion)

    return {
        'max_power_demand_kW': float(p_batt.max()),
        'max_power_absorption_kW': float(np.abs(p_batt.min())),
        'worst_case_power_peak_kW': float(p_batt.abs().max()),
        'min_capacity_excursion_kWh': float(max_excursion)
    }

def apply_signal_filters(df: pd.DataFrame, columns: List[str], method: Literal['savgol', 'butter', 'raw'] = 'savgol', **kwargs) -> pd.DataFrame:
    """Applies zero-phase digital filtering to simulate FC physical constraints."""
    df_filtered = df.copy()
    df_filtered.attrs['battery_specs'] = {}
    
    if method == 'raw':
        return df_filtered
        
    ignore_modes = kwargs.get('ignore_modes', None)
        
    for col in columns:
        if col not in df_filtered.columns:
            continue
            
        series = df_filtered[col]
        mask = series.isna()
        if mask.any():
            series = series.interpolate(method='linear')
            
        if method == 'savgol':
            window = kwargs.get('window', FILTERS.SAVGOL_DEFAULT['window'])
            polyorder = kwargs.get('polyorder', FILTERS.SAVGOL_DEFAULT['polyorder'])
            smoothed = savgol_filter(series, window_length=window, polyorder=polyorder)
            
        elif method == 'butter':
            order = kwargs.get('order', FILTERS.BUTTER_DEFAULT['order'])
            cutoff = kwargs.get('cutoff', FILTERS.BUTTER_DEFAULT['cutoff'])
            b, a = butter(order, cutoff, btype='low', analog=False)
            
            # Padlen protection for short data sequences
            padlen = min(3 * max(len(a), len(b)), len(series) - 1)
            smoothed = filtfilt(b, a, series, padlen=padlen) if len(series) > padlen else series.values
            
        smoothed_series = pd.Series(smoothed, index=df.index)
        smoothed_series[mask] = np.nan
        
        if col == 'AE_POWER(kW)':
            # Hard 1 MW limit as per MARINER scope
            smoothed_series = smoothed_series.clip(upper=1000.0, lower=0.0) 
            df_filtered.attrs['battery_specs'] = _compute_inline_battery_specs(df, smoothed_series, ignore_modes)
            
        df_filtered[col] = smoothed_series
            
    return df_filtered

def calc_derivatives_and_proxies(df: pd.DataFrame, dt_min: float) -> pd.DataFrame:
    """Calculates physical derivatives and electrical shock proxies."""
    new_cols = {}
    eps = PHYSICS.EPSILON
    
    if 'HEADING(degree)' in df.columns:
        rot = ((df['HEADING(degree)'].diff() + 180) % 360 - 180) / dt_min
        new_cols['ROT_DEG_PER_MIN'] = rot
        if 'AE_POWER(kW)' in df.columns:
            new_cols['MANEUVER_INTENSITY'] = rot.abs() * df['AE_POWER(kW)']

    if 'AE_POWER(kW)' in df.columns:
        tv_energy = (df['AE_POWER(kW)'].diff() / dt_min)**2
        new_cols['POWER_TV_ENERGY'] = tv_energy
        new_cols['REL_POWER_VOLATILITY'] = tv_energy / (df['AE_POWER(kW)']**2 + eps)

        if 'SHIP SPEED(knots)' in df.columns:
            new_cols['ENERGY_INTENSITY'] = df['AE_POWER(kW)'] / (df['SHIP SPEED(knots)'] + eps)
             
    # Power Factor Logic
    p_cols = ['GE162(kW)', 'GE262(kW)', 'GE362(kW)']
    pf_cols = ['GE164', 'GE264', 'GE364']
    
    if all(c in df.columns for c in p_cols + pf_cols) and 'AE_POWER(kW)' in df.columns:
        s_total = sum(df[p_cols[i]] / (df[pf_cols[i]] + eps) for i in range(3))
        pf_effective = (df['AE_POWER(kW)'] / (s_total + eps)).clip(0, 1)
        new_cols['PF_EFFECTIVE'] = pf_effective
        if 'POWER_TV_ENERGY' in new_cols:
             new_cols['VOLTAGE_STRESS'] = new_cols['POWER_TV_ENERGY'] * (1 - pf_effective)

    return df.assign(**new_cols)
    
def engineer_telemetry_features(raw_df: pd.DataFrame, filter_method: Literal['savgol', 'butter', 'raw'] = 'savgol', dropna: bool = True, **kwargs) -> pd.DataFrame:
    """Master functional pipeline for telemetry engineering."""
    if not isinstance(raw_df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex to compute dt.")
        
    dt_min = raw_df.index.to_series().diff().median().total_seconds() / 60.0
    cols_to_filter = ['AE_POWER(kW)', 'SHIP SPEED(knots)', 'HEADING_SIN', 'HEADING_COS']
    
    processed_df = (
        raw_df.copy()
        .pipe(calc_trig_headings)
        .pipe(apply_signal_filters, columns=cols_to_filter, method=filter_method, **kwargs)
        .pipe(calc_derivatives_and_proxies, dt_min=dt_min)
    )
    
    if dropna:
        crit_cols = [c for c in ['AE_POWER(kW)', 'MANEUVER_INTENSITY', 'POWER_TV_ENERGY'] if c in processed_df.columns]
        processed_df = processed_df.dropna(subset=crit_cols)
        
    return processed_df