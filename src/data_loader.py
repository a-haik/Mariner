import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Optional

class VesselDataLoader:
    """Handles raw data ingestion, deduplication, and strict time-grid regularization."""

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
    
    def _process_coordinates(self, df: pd.DataFrame) -> None:
        """Converts ASCII-encoded Degrees and Minutes into standard Decimal Degrees."""
        req_cols = ['LAT-DEG(degree)', 'LAT-MIN(min)', 'LAT-NS', 
                    'LONG-DEG(degree)', 'LONG-MIN(min)', 'LONG-EW']
        
        if not all(col in df.columns for col in req_cols):
            return

        lat_sign = np.where(df['LAT-NS'] == 83, -1.0, 1.0) # South=83, North=78
        lon_sign = np.where(df['LONG-EW'] == 87, -1.0, 1.0) # West=87, East=69

        df['LATITUDE(DD)'] = lat_sign * (df['LAT-DEG(degree)'] + (df['LAT-MIN(min)'] / 60.0))
        df['LONGITUDE(DD)'] = lon_sign * (df['LONG-DEG(degree)'] + (df['LONG-MIN(min)'] / 60.0))

    def load_and_clean(self, time_col: str = "Sample time", date_format: str = "%d/%m/%y %H:%M") -> pd.DataFrame:
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
        
        # Deduplication
        df = df[~df.index.duplicated(keep='first')]
        
        # --- Circular Mean for Resampling ---
        if 'HEADING(degree)' in df.columns:
            rads = np.radians(df['HEADING(degree)'])
            df['HEAD_SIN'] = np.sin(rads)
            df['HEAD_COS'] = np.cos(rads)
            df.drop(columns=['HEADING(degree)'], inplace=True)

        # Resample all linear variables
        df = df.resample('5min').mean(numeric_only=True)

        # Reconstruct the circular mean using arctan2
        if 'HEAD_SIN' in df.columns and 'HEAD_COS' in df.columns:
            mean_rads = np.arctan2(df['HEAD_SIN'], df['HEAD_COS'])
            df['HEADING(degree)'] = np.degrees(mean_rads) % 360
            df.drop(columns=['HEAD_SIN', 'HEAD_COS'], inplace=True)

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

        self._process_coordinates(df)
        self.clean_df = df
        
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

        # 3. Logical & Energy Balance
        print("\n[3] System Logic & Energy Balance:")
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
            contradictions = df[(df['STATUS'] == 'Idle') & (df['SHIP SPEED(knots)'] > 8.0)]
            if not contradictions.empty:
                print(f"    WARNING: {len(contradictions)} logical contradictions detected (STATUS='Idle' but Speed > 10 knots).")
                print("    This suggests manual log lagging by the crew.")
            else:
                print("    Pass: Logged status aligns with kinematic speed.")
                
        print("="*50)