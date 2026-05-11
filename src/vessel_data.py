import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union, List, Optional

class VesselDataProcessor:
    """
    A class for extracting, cleaning, and visualizing maritime load profiles.
    Designed for the MARINER data-to-testing pipeline.
    """

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

    def load_data(self, time_col: str = "Sample time", date_format: str = "%d/%m/%y %H:%M") -> 'VesselDataProcessor':
        if not self.file_path.exists():
            raise FileNotFoundError(f"Dataset not found at {self.file_path}")

        footer_line = self._find_footer_start()
        
        # Load only the telemetry data
        # We use nrows to stop exactly before the blank line/metadata
        read_params = {
            "filepath_or_buffer": self.file_path,
            "nrows": footer_line - 1 if footer_line > 0 else None,
            "engine": 'c',  # Faster C engine
            "na_values": ['', ' '],
            "parse_dates": [time_col],
            "date_format": date_format
        }

        self.df = pd.read_csv(**read_params)
        
        # Clean up: remove trailing commas if Pandas added 'Unnamed' columns
        self.clean_df = self.df.loc[:, ~self.df.columns.str.contains('^Unnamed')]
        
        # Set Index
        self.clean_df.set_index(time_col, inplace=True)
        self.clean_df.sort_index(inplace=True)

        print(f"Successfully loaded {len(self.clean_df)} telemetry rows.")
        return self

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

        ae_power=self.clean_df['AE_POWER(kW)']
        ge_sum=self.clean_df['GE162(kW)']+self.clean_df['GE262(kW)']+self.clean_df['GE362(kW)'] 

        P_offset = np.abs(ae_power - ge_sum)

        
            
        print("--- Data Sanity Check ---")
        print(f"Total rows: {len(self.clean_df)}")
        print(f"Missing values:\n{self.clean_df.isna().sum()}")
        print(f'Power balance check (AE_POWER - sum(GE)) mean: {P_offset.mean():.2f} kW, std: {P_offset.std():.2f} kW')

    def plot_series(self, 
                    columns: List[str], 
                    subplots: bool = False,
                    rolling_window: Optional[str] = None, 
                    secondary_y: Optional[str] = None,
                    **kwargs) -> tuple:
        """
        A flexible plotting tool for time-series data.
        
        Parameters:
        - columns: List of column names to plot.
        - subplots: If True, plots each column on a separate subplot.
        - rolling_window: e.g., '1h' for a 1-hour rolling mean overlay.
        - secondary_y: Column name to plot on a secondary y-axis (ignored if subplots=True).
        - **kwargs: Additional arguments passed to plt.subplots (e.g., figsize=(15, 8)).
        
        Returns:
        - fig, axes: The matplotlib Figure and Axes objects for further notebook customization.
        """
        if self.clean_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Default figsize if not provided
        kwargs.setdefault('figsize', (14, 4 * len(columns) if subplots else 6))
        
        fig, axes = plt.subplots(nrows=len(columns) if subplots else 1, ncols=1, **kwargs)
        
        # Ensure axes is iterable for consistency
        if not isinstance(axes, (list, np.ndarray)):
            axes = [axes]

        for i, col in enumerate(columns):
            if col not in self.clean_df.columns:
                print(f"Warning: Column '{col}' not found in clean_df.")
                continue
                
            ax = axes[i] if subplots else axes[0]
            
            # Handle secondary y-axis if not in subplot mode
            if not subplots and col == secondary_y:
                ax = ax.twinx()
            
            # Base signal
            ax.plot(self.clean_df.index, self.clean_df[col], alpha=0.6, label=f'{col} (Raw)')
            
            # Rolling average overlay
            if rolling_window:
                smoothed = self.clean_df[col].rolling(window=rolling_window, center=True).mean()
                ax.plot(self.clean_df.index, smoothed, linewidth=2, label=f'{col} (Mean {rolling_window})')
            
            ax.set_ylabel(col)
            if subplots or i == 0 or col == secondary_y:
                ax.legend(loc='upper left' if col != secondary_y else 'upper right')

        axes[-1].set_xlabel("Time")
        fig.suptitle(f"Time-Series Analysis: {', '.join(columns)}", y=1.02)
        fig.tight_layout()
        
        return fig, axes