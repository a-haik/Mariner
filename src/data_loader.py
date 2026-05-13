import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Optional

class VesselDataLoader:
    """Handles raw data ingestion, deduplication, and time-grid regularization."""

    def __init__(self, data_path: Union[str, Path]):
        self.file_path = Path(data_path)
        self.raw_df: Optional[pd.DataFrame] = None
        self.clean_df: Optional[pd.DataFrame] = None

    def _find_footer_start(self, delimiter: str = ",") -> int:
        with open(self.file_path, 'r') as f:
            for i, line in enumerate(f):
                if "Tag Name" in line or line.strip().startswith(delimiter * 5):
                    return i
        return -1 
    
    def _process_coordinates(self) -> None:
        """Converts ASCII-encoded Degrees and Minutes into standard Decimal Degrees."""
        req_cols = ['LAT-DEG(degree)', 'LAT-MIN(min)', 'LAT-NS', 
                    'LONG-DEG(degree)', 'LONG-MIN(min)', 'LONG-EW']
        
        if not all(col in self.clean_df.columns for col in req_cols):
            print("Warning: Missing geographic columns. Cannot process coordinates.")
            return

        lat_sign = np.where(self.clean_df['LAT-NS'] == 83, -1.0, 1.0) # South=83, North=78
        lon_sign = np.where(self.clean_df['LONG-EW'] == 87, -1.0, 1.0) # West=87, East=69

        self.clean_df['LATITUDE(DD)'] = lat_sign * (
            self.clean_df['LAT-DEG(degree)'] + (self.clean_df['LAT-MIN(min)'] / 60.0)
        )
        self.clean_df['LONGITUDE(DD)'] = lon_sign * (
            self.clean_df['LONG-DEG(degree)'] + (self.clean_df['LONG-MIN(min)'] / 60.0)
        )

    def _extract_features(self, dt_minutes: float = 5.0, epsilon: float = 1e-6) -> None:
        """Engineers physical and electrical shock proxies from telemetry."""
        df = self.clean_df
        
        # 1. Generator Activity & Transitions
        p_cols = ['GE162(kW)', 'GE262(kW)', 'GE362(kW)']
        if all(col in df.columns for col in p_cols):
            df['NUM_GENERATORS'] = (df[p_cols] > 5.0).sum(axis=1)
            df['GEN_TRANSITION'] = df['NUM_GENERATORS'].diff().fillna(0)
            
            # Load Sharing Symmetry Error
            active_gen_powers = df[p_cols].replace(0, np.nan) # Ignore inactive generators
            df['LOAD_SYMMETRY_ERROR'] = (active_gen_powers.max(axis=1) - active_gen_powers.min(axis=1)).fillna(0)

        # 2. Kinematic Features
        if 'HEADING(degree)' in df.columns:
            raw_diff = df['HEADING(degree)'].diff()
            df['ROT_DEG_PER_MIN'] = ((raw_diff + 180) % 360 - 180) / dt_minutes
            
            heading_rad = np.radians(df['HEADING(degree)'])
            df['HEADING_SIN'] = np.sin(heading_rad)
            df['HEADING_COS'] = np.cos(heading_rad)
            
        if 'AE_POWER(kW)' in df.columns and 'SHIP SPEED(knots)' in df.columns:
            df['ENERGY_INTENSITY'] = df['AE_POWER(kW)'] / (df['SHIP SPEED(knots)'] + epsilon)
            # Power Total Variation (TV) Energy
            df['POWER_TV_ENERGY'] = (df['AE_POWER(kW)'].diff() / dt_minutes)**2

        # 3. Electrical Shock Proxies (Power Factor)
        pf_cols = ['GE164', 'GE264', 'GE364']
        if all(col in df.columns for col in pf_cols + p_cols) and 'AE_POWER(kW)' in df.columns:
            # Calculate total apparent power S = P / PF
            S_total = np.zeros(len(df))
            for p_col, pf_col in zip(p_cols, pf_cols):
                # Add to apparent power only if PF > 0 to avoid ZeroDivisionError
                S_total += np.where(df[pf_col] > epsilon, df[p_col] / df[pf_col], 0)
                
            df['PF_EFFECTIVE'] = np.where(S_total > epsilon, df['AE_POWER(kW)'] / S_total, 1.0)
            df['PF_DERIVATIVE'] = df['PF_EFFECTIVE'].diff() / dt_minutes

    def load_and_clean(self, time_col: str = "Sample time", date_format: str = "%d/%m/%y %H:%M") -> pd.DataFrame:
        """The master execution method for the data pipeline."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"Dataset not found at {self.file_path}")

        footer_line = self._find_footer_start()
        
        read_params = {
            "filepath_or_buffer": self.file_path,
            "nrows": footer_line - 1 if footer_line > 0 else None,
            "engine": 'c',
            "na_values": ['', ' ', 'NaN', 'null'],
            "parse_dates": [time_col],
            "date_format": date_format
        }

        self.raw_df = pd.read_csv(**read_params)
        
        if "Unnamed" in self.raw_df.columns[-1]:
            self.raw_df.rename(columns={self.raw_df.columns[-1]: "STATUS"}, inplace=True)
        
        df = self.raw_df.loc[:, ~self.raw_df.columns.str.contains('^Unnamed')].copy()
        df.set_index(time_col, inplace=True)
        df.sort_index(inplace=True)
        
        # Deduplication & Resampling
        df = df[~df.index.duplicated(keep='first')]
        df = df.resample('5min').mean(numeric_only=True)
        
        # Interpolation & Bounding
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].interpolate(method='time', limit=3)
        
        power_current_cols = [col for col in df.columns if 'kW' in col or '(A)' in col]
        for col in power_current_cols:
            df[col] = df[col].clip(lower=0.0)

        # Restore Status Column
        if 'STATUS' in self.raw_df.columns:
            status_series = self.raw_df.set_index(time_col)[~self.raw_df.set_index(time_col).index.duplicated(keep='first')]['STATUS']
            df['STATUS'] = status_series.reindex(df.index).ffill(limit=3)

        self.clean_df = df
        
        # Execute internal processing
        self._process_coordinates()
        self._extract_features()

        print(f"Successfully loaded, cleaned, and processed {len(self.clean_df)} telemetry rows.")
        return self.clean_df
    
    def sanity_check(self) -> None:
        """
        Executes a rigorous engineering sanity check on the cleaned telemetry,
        flagging data gaps, physical boundary violations, and logical anomalies.
        """
        df = self.clean_df
        if df is None or df.empty:
            print("ERROR: No data loaded or dataframe is empty.")
            return

        print("="*50)
        print(f"TELEMETRY SANITY CHECK REPORT: {self.file_path.name}")
        print("="*50)

        # 1. Basic Completeness
        print("[1] Basic Metrics:")
        print(f"    Total Rows: {len(df)}")
        total_nans = df.isna().sum().sum()
        if total_nans > 0:
            print(f"    WARNING: {total_nans} missing values detected across the dataframe.")
            print(df.isna().sum()[df.isna().sum() > 0].to_string())
        else:
            print("    Pass: No missing values detected.")

        # 2. Time-Grid Continuity
        print("\n[2] Time-Grid Continuity:")
        # Check if the index is strictly increasing
        is_monotonic = df.index.is_monotonic_increasing
        print(f"    Strictly Monotonic Time Index: {'Pass' if is_monotonic else 'FAIL'}")
        
        # Calculate time gaps
        time_diffs = df.index.to_series().diff()
        expected_dt = pd.Timedelta(minutes=5)
        large_gaps = time_diffs[time_diffs > expected_dt]
        if not large_gaps.empty:
            print(f"    WARNING: {len(large_gaps)} time gaps larger than 5 minutes detected.")
            print(f"    Largest gap: {large_gaps.max()}")
        else:
            print("    Pass: No significant time gaps detected.")

        # 3. Physical Boundary Violations
        print("\n[3] Physical Boundary Checks:")
        violations = 0
        
        if 'PF_EFFECTIVE' in df.columns:
            pf_max = df['PF_EFFECTIVE'].max()
            if pf_max > 1.01: # Small tolerance for floating point math
                print(f"    FAIL: Effective Power Factor exceeds 1.0 (Max: {pf_max:.3f})")
                violations += 1
                
        if 'SHIP SPEED(knots)' in df.columns:
            speed_max = df['SHIP SPEED(knots)'].max()
            speed_min = df['SHIP SPEED(knots)'].min()
            if speed_max > 25.0: # Unrealistic for a Handymax tanker
                print(f"    WARNING: Suspiciously high speed detected (Max: {speed_max:.1f} knots)")
                violations += 1
            if speed_min < -0.1:
                print(f"    FAIL: Negative speed detected (Min: {speed_min:.1f} knots)")
                violations += 1

        if violations == 0:
            print("    Pass: All engineered physical features within realistic bounds.")

        # 4. Logical & Energy Balance
        print("\n[4] System Logic & Energy Balance:")
        p_cols = ['GE162(kW)', 'GE262(kW)', 'GE362(kW)']
        if 'AE_POWER(kW)' in df.columns and all(c in df.columns for c in p_cols):
            ge_sum = df[p_cols].sum(axis=1)
            p_offset_abs = np.abs(df['AE_POWER(kW)'] - ge_sum)
            mean_error = p_offset_abs.mean()
            if mean_error > 50.0: # Arbitrary threshold, adjust based on sensor accuracy
                print(f"    WARNING: High mean power imbalance between AE and Generators: {mean_error:.2f} kW")
            else:
                print(f"    Pass: Mean power balance within acceptable limits ({mean_error:.2f} kW).")

        # Logical contradiction: Ship is idle but moving fast
        if 'STATUS' in df.columns and 'SHIP SPEED(knots)' in df.columns:
            contradictions = df[(df['STATUS'] == 'Idle') & (df['SHIP SPEED(knots)'] > 10.0)]
            if not contradictions.empty:
                print(f"    WARNING: {len(contradictions)} logical contradictions detected (STATUS='Idle' but Speed > 10 knots).")
                print("    This suggests manual log lagging by the crew.")
            else:
                print("    Pass: Logged status aligns with kinematic speed.")
                
        print("="*50)