import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union, List, Optional

class VesselDataProcessor:
    """
    A class for extractingand visualizing maritime load profiles
    """

    def __init__(self, data_path: Union[str, Path]):
        self.file_path = Path(data_path)
        self.df = None
        self.processed_df = None

    def load_data(self, time_col:str = "Sample time", **kwargs) -> 'VesselDataProcessor':
        """
        Load data from a CSV file and set the time column as the index.
        
        Parameters:
        time_column (str): The name of the column to be used as the time index.
        **kwargs: Additional keyword arguments to pass to VesselDataProcessor.
        
        Returns:
        VesselDataProcessor: The instance of the class for method chaining.
        """

        if not self.file_path.exists():
            raise FileNotFoundError(f"Cannot find dataset at {self.file_path}")
            
        # Load raw data
        self.df = pd.read_csv(self.file_path, **kwargs)
        
        # Enforce time-series fundamentals
        if time_col in self.df.columns:
            self.df[time_col] = pd.to_datetime(self.df[time_col])
            self.df.set_index(time_col, inplace=True)
            self.df.sort_index(inplace=True)
        else:
            raise KeyError(f"Time column '{time_col}' not found in dataset.")
            
        # Initially, clean_df is just a copy of df
        self.clean_df = self.df.copy()
        return self
    
    def sanity_check(self) -> None:
        """
        Prints a quick overview of missing values and physical anomalies.
        """
        if self.clean_df is None:
            print("No data loaded.")
            return
            
        print("--- Data Sanity Check ---")
        print(f"Total rows: {len(self.clean_df)}")
        print(f"Missing values:\n{self.clean_df.isna().sum()}")
        # We will add physical checks here later

    def plot_series(self, columns: List[str], 
                    rolling_window: Optional[str] = None, 
                    secondary_y: Optional[str] = None) -> None:
        """
        A flexible plotting tool capable of handling multiple signals and secondary axes.
        """
        if self.clean_df is None:
            raise ValueError("Data not loaded.")

        fig, ax1 = plt.subplots(figsize=(14, 6))
        
        for col in columns:
            if col not in self.clean_df.columns:
                print(f"Warning: Column '{col}' not found.")
                continue
                
            # Plot on secondary axis if specified
            ax = ax1.twinx() if col == secondary_y else ax1
            
            # Base signal
            ax.plot(self.clean_df.index, self.clean_df[col], alpha=0.6, label=f'{col} (Raw)')
            
            # Optional rolling average overlay
            if rolling_window:
                smoothed = self.clean_df[col].rolling(window=rolling_window, center=True).mean()
                ax.plot(self.clean_df.index, smoothed, linewidth=2, label=f'{col} (Mean {rolling_window})')
                
            ax.set_ylabel(col)
            ax.legend(loc='upper left' if ax == ax1 else 'upper right')

        ax1.set_xlabel("Time")
        ax1.set_title(f"Time-Series Analysis: {', '.join(columns)}")
        plt.tight_layout()
        plt.show() 