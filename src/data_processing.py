import pandas as pd
import numpy as np
from scipy.signal import savgol_filter, butter, filtfilt
from typing import Literal

class DataProcessor:
    """
    Acts as the Physics/Feature layer. 
    Applies signal processing and derives mission-critical metrics for PEMFC AST.
    """
    
    def __init__(self, df: pd.DataFrame, dt_minutes: float = 5.0):
        # We take the output of VesselDataLoader.load_and_clean() as the input here
        self.df = df.copy()
        self.dt = dt_minutes
        self.epsilon = 1e-6
        
        # Define the continuous signals that are mathematically safe to filter
        self.continuous_signals = ['AE_POWER(kW)', 'SHIP SPEED(knots)']

    def extract_features_and_filter(self, 
                                 filter_method: Literal['savgol', 'butterworth', 'raw'] = 'savgol',
                                 window_size: int = 5,
                                 dropna: bool = True,
                                 **kwargs) -> pd.DataFrame:
        """
        Master method: Filters raw signals and derives all kinematic/electrical 
        features in a single consistent pass.
        """
        # 1. Base Kinematics (Must happen before filtering if we filter trig headings)
        self._derive_base_kinematics()
        
        # 2. Apply Filtering to Continuous Signals (including the newly derived trig headings)
        signals_to_filter = self.continuous_signals + ['HEADING_SIN', 'HEADING_COS']
        for col in signals_to_filter:
            self.df[col] = self.apply_filter(col, filter_method, window_size=window_size, **kwargs)
        
        # 3. Derive Electrical Shock Proxies and Rate of Turn (Uses Filtered Data)
        self._derive_derivatives_and_proxies()
        
        if dropna:
            critical_cols = [
                'AE_POWER(kW)', 'NUM_GENERATORS', 'ENERGY_INTENSITY', 
                'MANEUVER_INTENSITY', 'POWER_TV_ENERGY', 'PF_DERIVATIVE'
            ]
            # Only drop based on the columns that actually exist to avoid KeyErrors
            existing_cols = [c for c in critical_cols if c in self.df.columns]
            self.df.dropna(subset=existing_cols, inplace=True)
            
        return self.df

    def apply_filter(self, column: str, method: str = 'savgol', **kwargs) -> pd.Series:
        """
        Applies zero-phase filtering.
        method: 'savgol', 'butter', or 'raw'
        kwargs: 
            savgol -> window (int), polyorder (int)
            butter -> order (int), cutoff (float, normalized 0 to 1)
        """
        if column not in self.df.columns or method == 'raw':
            return self.df[column] if column in self.df.columns else pd.Series(np.nan, index=self.df.index)
            
        series = self.df[column]
        mask = series.isna()
        if mask.any():
            series = series.interpolate(method='linear')
            
        if method == 'savgol':
            window = kwargs.get('window', 5)
            polyorder = kwargs.get('polyorder', 2)
            smoothed = savgol_filter(series, window_length=window, polyorder=polyorder)
            
        elif method == 'butter':
            order = kwargs.get('order', 2)
            cutoff = kwargs.get('cutoff', 0.1) # Normalized frequency
            b, a = butter(order, cutoff, btype='low', analog=False)
            smoothed = filtfilt(b, a, series) # Zero-phase filtering
            
        else:
            raise ValueError(f"Unknown filter method: {method}")

        smoothed_series = pd.Series(smoothed, index=self.df.index)
        smoothed_series[mask] = np.nan
        return smoothed_series

    def _derive_base_kinematics(self) -> None:
        """Calculates Trig encodings for HMM/Clustering before filtering."""
        if 'HEADING(degree)' in self.df.columns:
            rads = np.radians(self.df['HEADING(degree)'])
            self.df['HEADING_SIN'] = np.sin(rads)
            self.df['HEADING_COS'] = np.cos(rads)

    def _derive_derivatives_and_proxies(self) -> None:
        """Calculates derivatives (RoT, TV) and Electrical Shock Proxies on filtered data."""
        # Rate of Turn
        if 'HEADING(degree)' in self.df.columns:
            raw_diff = self.df['HEADING(degree)'].diff()
            self.df['ROT_DEG_PER_MIN'] = ((raw_diff + 180) % 360 - 180) / self.dt
            self.df['MANEUVER_INTENSITY'] = self.df['ROT_DEG_PER_MIN'].abs() * self.df['AE_POWER(kW)']
            self.df['LATERAL_ACCELERATION'] = self.df['ROT_DEG_PER_MIN'] * self.df['SHIP SPEED(knots)']

        # Energy Intensity and TV Energy
        if 'AE_POWER(kW)' in self.df.columns:
            self.df['POWER_TV_ENERGY'] = (self.df['AE_POWER(kW)'].diff() / self.dt)**2
            self.df['REL_POWER_VOLATILITY'] = self.df['POWER_TV_ENERGY'] / (self.df['AE_POWER(kW)']**2 + self.epsilon)
            if 'SHIP SPEED(knots)' in self.df.columns:
                self.df['ENERGY_INTENSITY'] = self.df['AE_POWER(kW)'] / (self.df['SHIP SPEED(knots)'] + self.epsilon)
            

        # Power Factor Logic
        p_cols = ['GE162(kW)', 'GE262(kW)', 'GE362(kW)']
        pf_cols = ['GE164', 'GE264', 'GE364']
        
        if all(c in self.df.columns for c in p_cols + pf_cols) and 'AE_POWER(kW)' in self.df.columns:
            s_total = sum(self.df[p_cols[i]] / (self.df[pf_cols[i]] + self.epsilon) for i in range(3))
            self.df['PF_EFFECTIVE'] = (self.df['AE_POWER(kW)'] / (s_total + self.epsilon)).clip(0, 1)
            self.df['PF_DERIVATIVE'] = self.df['PF_EFFECTIVE'].diff() / self.dt
            self.df['VOLTAGE_STRESS'] = self.df['POWER_TV_ENERGY'] * (1 - self.df['PF_EFFECTIVE'])
            
        # Discrete Generator Logic (NEVER FILTERED)
        if all(c in self.df.columns for c in p_cols):

            self.df['NUM_GENERATORS'] = (self.df[p_cols] > 5.0).sum(axis=1)
            self.df['GEN_TRANSITION'] = self.df['NUM_GENERATORS'].diff().fillna(0)