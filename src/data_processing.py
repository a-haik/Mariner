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

def apply_signal_filters(
    df: pd.DataFrame, 
    columns: List[str], 
    method: Literal['savgol', 'butter', 'raw'] = 'savgol',
    **kwargs
) -> pd.DataFrame:
    """Applies zero-phase digital filtering to specified columns."""
    if method == 'raw':
        return df
        
    df_filtered = df.copy()
    
    for col in columns:
        if col not in df_filtered.columns:
            continue
            
        series = df_filtered[col]
        mask = series.isna()
        
        # Temporarily interpolate NaNs for filter stability
        if mask.any():
            series = series.interpolate(method='linear')
            
        if method == 'savgol':
            window = kwargs.get('window', 5)
            polyorder = kwargs.get('polyorder', 2)
            smoothed = savgol_filter(series, window_length=window, polyorder=polyorder)
            
        elif method == 'butter':
            order = kwargs.get('order', 2)
            cutoff = kwargs.get('cutoff', 0.1)
            b, a = butter(order, cutoff, btype='low', analog=False)
            smoothed = filtfilt(b, a, series)
            
        # Restore NaNs to avoid hallucinating data
        smoothed_series = pd.Series(smoothed, index=df.index)
        smoothed_series[mask] = np.nan
        df_filtered[col] = smoothed_series
        
    return df_filtered

def calc_derivatives_and_proxies(df: pd.DataFrame, dt_min: float) -> pd.DataFrame:
    """Calculates physical derivatives and electrical shock proxies."""
    new_cols = {}
    
    # 1. Rate of Turn & Kinematics
    if 'HEADING(degree)' in df.columns:
        raw_diff = df['HEADING(degree)'].diff()
        rot = ((raw_diff + 180) % 360 - 180) / dt_min
        new_cols['ROT_DEG_PER_MIN'] = rot
        
        if 'AE_POWER(kW)' in df.columns:
            new_cols['MANEUVER_INTENSITY'] = rot.abs() * df['AE_POWER(kW)']

    # 2. Energy Intensity & Volatility
    if 'AE_POWER(kW)' in df.columns:
        tv_energy = (df['AE_POWER(kW)'].diff() / dt_min)**2
        new_cols['POWER_TV_ENERGY'] = tv_energy
        # Use PHYSICS.EPSILON from config
        new_cols['REL_POWER_VOLATILITY'] = tv_energy / (df['AE_POWER(kW)']**2 + PHYSICS.EPSILON)

        
        if 'SHIP SPEED(knots)' in df.columns:
            new_cols['ENERGY_INTENSITY'] = df['AE_POWER(kW)'] / (df['SHIP SPEED(knots)'] + PHYSICS.EPSILON)
            
    # 3. Power Factor Logic
    p_cols = ['GE162(kW)', 'GE262(kW)', 'GE362(kW)']
    pf_cols = ['GE164', 'GE264', 'GE364']
    
    if all(c in df.columns for c in p_cols + pf_cols) and 'AE_POWER(kW)' in df.columns:
        s_total = sum(df[p_cols[i]] / (df[pf_cols[i]] + PHYSICS.EPSILON) for i in range(3))
        pf_effective = (df['AE_POWER(kW)'] / (s_total + PHYSICS.EPSILON)).clip(0, 1)
        
        new_cols['PF_EFFECTIVE'] = pf_effective
        new_cols['PF_DERIVATIVE'] = pf_effective.diff() / dt_min
        
        if 'POWER_TV_ENERGY' in new_cols:
             new_cols['VOLTAGE_STRESS'] = new_cols['POWER_TV_ENERGY'] * (1 - pf_effective)
             
    # 4. Discrete Generator Logic
    if all(c in df.columns for c in p_cols):
        num_gens = (df[p_cols] > 5.0).sum(axis=1)
        new_cols['NUM_GENERATORS'] = num_gens
        new_cols['GEN_TRANSITION'] = num_gens.diff().fillna(0)

    return df.assign(**new_cols)
    
def engineer_telemetry_features(
    raw_df: pd.DataFrame, 
    filter_method: Literal['savgol', 'butter', 'raw'] = 'savgol',
    dropna: bool = True
) -> pd.DataFrame:
    """Master functional pipeline for telemetry engineering."""
    
    # 1. Dynamically calculate dt_min using the median time gap
    if not isinstance(raw_df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex to compute dt.")
        
    median_dt_seconds = raw_df.index.to_series().diff().median().total_seconds()
    dt_min = median_dt_seconds / 60.0

    # 2. Retrieve appropriate filter kwargs from config
    filter_kwargs = FILTERS.SAVGOL_DEFAULT if filter_method == 'savgol' else FILTERS.BUTTER_DEFAULT
    cols_to_filter = ['AE_POWER(kW)', 'SHIP SPEED(knots)', 'HEADING_SIN', 'HEADING_COS']
    
    # 3. Execute the functional pipeline
    processed_df = (
        raw_df.copy()
        .pipe(calc_trig_headings)
        .pipe(apply_signal_filters, columns=cols_to_filter, method=filter_method, **filter_kwargs)
        .pipe(calc_derivatives_and_proxies, dt_min=dt_min)
    )
    
    if dropna:
        critical_cols = [
            'AE_POWER(kW)', 'NUM_GENERATORS', 'MANEUVER_INTENSITY', 'POWER_TV_ENERGY'
        ]
        existing_cols = [c for c in critical_cols if c in processed_df.columns]
        processed_df = processed_df.dropna(subset=existing_cols)
        
    return processed_df