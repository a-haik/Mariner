import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Optional

class VesselDataLoader:
    """Handles raw data ingestion, deduplication, and time-grid regularization."""

    def __init__(self, data_path: Union[str, Path]):
        self.file_path = Path(data_path)
        self.df: Optional[pd.DataFrame] = None
        self.clean_df: Optional[pd.DataFrame] = None
        self.metadata: Optional[pd.DataFrame] = None

    def _find_footer_start(self, delimiter: str = ",") -> int:
        """
        Scans the file to find the line number where metadata begins.
        In your case, look for 'Tag Name' or a string of empty commas.
        """
        with open(self.file_path, 'r') as f:
            for i, line in enumerate(f):
                # Check for the specific header of your metadata footer
                if "Tag Name" in line or line.strip().startswith(delimiter * 5):
                    return i
        return -1 # No footer found
    
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

    def _extract_features(self, ge_threshold_kw: float = 5.0) -> None:
        """Engineers new features required for the ML classification layer."""
        ge_cols = ['GE162(kW)', 'GE262(kW)', 'GE362(kW)']
        if all(col in self.clean_df.columns for col in ge_cols):
            # Calculate active generators based on power output threshold
            self.clean_df['NUM_GENERATORS'] = (self.clean_df[ge_cols] > ge_threshold_kw).sum(axis=1)

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

    def extract_metadata(self) -> pd.DataFrame:
        """
        Optional: Implement this to capture the definitions footer 
        into a separate dataframe for feature mapping.
        """
        footer_line = self._find_footer_start()
        if footer_line == -1:
            return pd.DataFrame()
            
        return pd.read_csv(self.file_path, skiprows=footer_line)
    
    def sanity_check(self) -> None:
        """
        Prints a quick overview of missing values and physical anomalies.
        """
        if self.clean_df is None:
            print("No data loaded.")
            return

        if 'AE_POWER(kW)' in self.clean_df.columns:
            ae_power=self.clean_df['AE_POWER(kW)']
        if 'GE162(kW)' in self.clean_df.columns:
            ge12=self.clean_df['GE162(kW)']
        if 'GE262(kW)' in self.clean_df.columns:
            ge22=self.clean_df['GE262(kW)']
        if 'GE362(kW)' in self.clean_df.columns:
            ge32=self.clean_df['GE362(kW)']

        ge_sum=ge12+ge22+ge32
        P_offset_abs = np.abs(ae_power - ge_sum)

        print("--- Data Sanity Check ---")
        print(f"Total rows: {len(self.clean_df)}")
        print(f"Missing values:\n{self.clean_df.isna().sum()}")
        print(f'Power balance check (AE_POWER - sum(GE)) mean: {P_offset_abs.mean():.2f} kW, std: {P_offset_abs.std():.2f} kW')