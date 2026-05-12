import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

        # Fix the Unnamed STATUS Column:
        if "Unnamed" in self.df.columns[-1]:
            self.df.rename(columns={self.df.columns[-1]: "STATUS"}, inplace=True)
        
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

    def plot_series(self, 
                    columns: List[str], 
                    subplots: bool = False,
                    rolling_window: Optional[str] = None, 
                    secondary_y: Optional[str] = None,
                    show_status: bool = True,
                    **kwargs) -> tuple:
        """
        A flexible plotting tool for time-series data.
        
        Parameters:
        - columns: List of column names to plot.
        - subplots: If True, plots each column on a separate subplot.
        - rolling_window: e.g., '1h' for a 1-hour rolling mean overlay.
        - secondary_y: Column name to plot on a secondary y-axis (ignored if subplots=True).
        - show_status: If True, adds background color blocks based on vessel status.
        - **kwargs: Additional arguments passed to plt.subplots (e.g., figsize=(15, 8)).
        
        Returns:
        - fig, axes: The matplotlib Figure and Axes objects for further notebook customization.
        """
        if self.clean_df is None:
            raise ValueError("Data not loaded.")

        status_colors = {
            'idle': '#f0f0f0', 'laden': '#d1e7dd', 'ballast': '#fff3cd', 
            'discharging': '#f8d7da', 'loading': '#cfe2ff'
        }

        kwargs.setdefault('figsize', (14, 4 * len(columns) if subplots else 7))
        fig, axes = plt.subplots(nrows=len(columns) if subplots else 1, ncols=1, **kwargs)
        
        if not isinstance(axes, (list, np.ndarray)):
            axes = [axes]

        # 1. Color Management: Use the standard cycle but track it manually
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        
        # 2. Extract Status Transitions (same logic as yours)
        status_col = 'STATUS'
        if show_status and status_col in self.clean_df.columns:
            status_changes = (self.clean_df[status_col] != self.clean_df[status_col].shift()).fillna(True)
            change_indices = self.clean_df.index[status_changes].tolist()
            change_indices.append(self.clean_df.index[-1])

        # Track handles and labels for the unified legend
        all_handles = []
        all_labels = []

        for i, col in enumerate(columns):
            if col not in self.clean_df.columns:
                continue
                
            ax = axes[i] if subplots else axes[0]
            
            # Determine if we need a twin axis
            is_secondary = (not subplots and col == secondary_y)
            plot_ax = ax.twinx() if is_secondary else ax
            
            # Use the index i to pick a unique color from the cycle
            color = colors[i % len(colors)]
            
            # Plot Raw
            line, = plot_ax.plot(self.clean_df.index, self.clean_df[col], 
                                 alpha=0.8, color=color, label=f'{col} (Raw)')
            all_handles.append(line)
            all_labels.append(f'{col} (Raw)')
            
            if rolling_window:
                smoothed = self.clean_df[col].rolling(window=rolling_window, center=True).mean()
                s_line, = plot_ax.plot(self.clean_df.index, smoothed, 
                                       linewidth=2, color=color, linestyle='--', 
                                       label=f'{col} (Mean {rolling_window})')
                all_handles.append(s_line)
                all_labels.append(f'{col} (Mean {rolling_window})')

            # 3. Add Background Color Blocks (Only once if subplots=False)
            if show_status and status_col in self.clean_df.columns:
                if subplots or i == 0: # Avoid drawing spans multiple times on the same ax
                    for start, end in zip(change_indices[:-1], change_indices[1:]):
                        current_status = self.clean_df.loc[start, status_col]
                        bg_color = status_colors.get(str(current_status).lower(), 'white')
                        ax.axvspan(start, end, color=bg_color, alpha=0.3, zorder=-1)

            plot_ax.set_ylabel(col, color=color if is_secondary else 'black')

        # 4. Unified Legend Logic
        if subplots:
            for ax in axes:
                ax.legend(loc='upper left')
        else:
            # Combine all handles from ax and twin_ax into one legend on the first ax
            axes[0].legend(all_handles, all_labels, loc='upper left')

        if show_status and status_col in self.clean_df.columns:
            patches = [mpatches.Patch(color=color, label=status.capitalize(), alpha=0.3) 
                       for status, color in status_colors.items() 
                       if status in self.clean_df[status_col].str.lower().unique()]
            fig.legend(handles=patches, title="Vessel Status", loc='center left', bbox_to_anchor=(1, 0.5))

        axes[-1].set_xlabel("Time")
        fig.suptitle(f"Time-Series Analysis: {', '.join(columns)}", y=1.02)
        fig.tight_layout()
            
        return fig, axes