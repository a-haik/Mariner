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
    Deduces battery properties by isolating continuous voyage blocks.
    Resets the energy integration tracking between decoupled phases to eliminate
    artificial timeline splicing artifacts and simulate system state resets.
    """
    raw_power = df['AE_POWER(kW)']
    p_batt = raw_power - filtered_power
    
    # 1. Apply dynamic operational mode masking
    if ignore_modes and 'MODE' in df.columns:
        ignored_clean = [str(m).lower() for m in ignore_modes]
        status_mask = df['MODE'].astype(str).str.lower().isin(ignored_clean)
        p_batt = p_batt.mask(status_mask, 0.0)
        
    median_dt_sec = raw_power.index.to_series().diff().median().total_seconds()
    dt_hours = median_dt_sec / 3600.0

    # 2. Determine independent operational blocks
    # If stay_id isn't present in the dataframe yet, we generate a localized continuous block ID
    if 'stay_id' in df.columns:
        block_ids = df['stay_id']
    else:
        # Generate a fallback block ID based on status changes
        status_series = df['STATUS'] if 'STATUS' in df.columns else pd.Series('SingleBlock', index=df.index)
        block_ids = (status_series != status_series.shift()).cumsum()

    max_excursion_across_blocks = 0.0
    
    # 3. Loop through each independent operational block to isolate integrations
    for b_id, group_idx in p_batt.groupby(block_ids).groups.items():
        p_batt_block = p_batt.loc[group_idx]
        
        if p_batt_block.empty or len(p_batt_block) < 2:
            continue
            
        # Center the power within this specific isolated phase block to model local EMS balance
        p_batt_centered = p_batt_block - p_batt_block.mean()
        e_cumulative = (p_batt_centered * dt_hours).cumsum()
        
        # Track the maximum energy capacity swing encountered in any single continuous operation
        block_excursion = e_cumulative.max() - e_cumulative.min()
        if block_excursion > max_excursion_across_blocks:
            max_excursion_across_blocks = block_excursion

    return {
        'max_power_demand_kW': float(p_batt.max()),
        'max_power_absorption_kW': float(np.abs(p_batt.min())),
        'worst_case_power_peak_kW': float(p_batt.abs().max()),
        'min_capacity_excursion_kWh': float(max_excursion_across_blocks)
    }

def apply_signal_filters(
    df: pd.DataFrame, 
    columns: List[str], 
    method: Literal['savgol', 'butter', 'raw'] = 'savgol',
    **kwargs
) -> pd.DataFrame:
    """Applies zero-phase digital filtering, clips at a physical 1MW constraint, and hooks inline battery metrics."""
    df_filtered = df.copy()
    df_filtered.attrs['battery_specs'] = {}
    
    if method == 'raw':
        return df_filtered
        
    # Extract the custom exclusion modes parameter if provided during the parameter sweep cell
    ignore_modes = kwargs.get('ignore_modes', None)
        
    for col in columns:
        if col not in df_filtered.columns:
            continue
            
        series = df_filtered[col]
        mask = series.isna()
        
        if mask.any():
            series = series.interpolate(method='linear')
            
        if method == 'savgol':
            window = kwargs.get('window', 10)
            polyorder = kwargs.get('polyorder', 2)
            smoothed = savgol_filter(series, window_length=window, polyorder=polyorder)
            
        elif method == 'butter':
            order = kwargs.get('order', 2)
            cutoff = kwargs.get('cutoff', 0.05)
            b, a = butter(order, cutoff, btype='low', analog=False)
            smoothed = filtfilt(b, a, series)
            
        smoothed_series = pd.Series(smoothed, index=df.index)
        smoothed_series[mask] = np.nan
        
        # Enforce the physical 1 MW plant constraint
        if col == 'AE_POWER(kW)':
            smoothed_series = smoothed_series.clip(upper=1000.0)
            
        df_filtered[col] = smoothed_series
        
        # Inline hook execution with forwarded exclusion tracking array
        if col == 'AE_POWER(kW)':
            df_filtered.attrs['battery_specs'] = _compute_inline_battery_specs(
                df=df, 
                filtered_power=smoothed_series,
                ignore_modes=ignore_modes
            )
            
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
    dropna: bool = True,
    **kwargs
) -> pd.DataFrame:
    """Master functional pipeline for telemetry engineering with dynamic filter parameter forwarding."""
    
    # 1. Dynamically calculate dt_min using the median time gap
    if not isinstance(raw_df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex to compute dt.")
        
    median_dt_seconds = raw_df.index.to_series().diff().median().total_seconds()
    dt_min = median_dt_seconds / 60.0

    # 2. Retrieve appropriate filter baseline kwargs from config and overlay runtime overrides
    if filter_method == 'savgol':
        filter_kwargs = FILTERS.SAVGOL_DEFAULT.copy()
    else:
        filter_kwargs = FILTERS.BUTTER_DEFAULT.copy()
        
    # Intercept and update with any explicitly passed options (e.g., cutoff=fc or window=w)
    filter_kwargs.update(kwargs)
    
    cols_to_filter = ['AE_POWER(kW)', 'SHIP SPEED(knots)', 'HEADING_SIN', 'HEADING_COS']
    
    # 3. Execute the functional pipeline with custom parameter dictionaries
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