import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Union, List, Optional
import folium
import branca.colormap as cm
import math

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
    
    def process_coordinates(self) -> 'VesselDataProcessor':
        """
        Converts ASCII-encoded Degrees and Minutes into standard Decimal Degrees (DD).
        Adds 'LATITUDE(DD)' and 'LONGITUDE(DD)' columns to clean_df.
        """
        if self.clean_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        req_cols = ['LAT-DEG(degree)', 'LAT-MIN(min)', 'LAT-NS', 
                    'LONG-DEG(degree)', 'LONG-MIN(min)', 'LONG-EW']
        
        if not all(col in self.clean_df.columns for col in req_cols):
            print("Warning: Missing geographic columns. Cannot process coordinates.")
            return self

        # Map ASCII values to algebraic signs
        # North (78) -> +1, South (83) -> -1
        lat_sign = np.where(self.clean_df['LAT-NS'] == 83, -1.0, 1.0)
        
        # East (69) -> +1, West (87) -> -1
        lon_sign = np.where(self.clean_df['LONG-EW'] == 87, -1.0, 1.0)

        # Compute Decimal Degrees
        self.clean_df['LATITUDE(DD)'] = lat_sign * (
            self.clean_df['LAT-DEG(degree)'] + (self.clean_df['LAT-MIN(min)'] / 60.0)
        )
        self.clean_df['LONGITUDE(DD)'] = lon_sign * (
            self.clean_df['LONG-DEG(degree)'] + (self.clean_df['LONG-MIN(min)'] / 60.0)
        )

        print("Successfully processed standard Latitude and Longitude.")
        return self
    

    @staticmethod
    def _calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculates the initial bearing (forward azimuth) between two coordinates."""
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlon = lon2 - lon1
        x = math.sin(dlon) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(dlon))
        initial_bearing = math.atan2(x, y)
        return math.degrees(initial_bearing)
    
    def _get_arrow_vertices(self, lat: float, lon: float, bearing_deg: float, size: float = 0.005) -> List[List[float]]:
        """
        Generates vertices for an arrowhead polygon.
        Defined in map units (degrees) so it scales with zoom.
        """
        # THE FIX: Convert degrees to radians for Python's math functions!
        bearing_rad = math.radians(bearing_deg)
        
        # Rotate vertices by the bearing
        cos_b = math.cos(bearing_rad)
        sin_b = math.sin(bearing_rad)
        
        # Adjust the x-offset (longitude) by the cosine of latitude 
        # to maintain the shape aspect ratio on a Mercator projection
        lon_scale = 1.0 / math.cos(math.radians(lat))
        
        # Vertices in local (y=North, x=East) coordinates
        # Tip, bottom-left, bottom-right
        local_coords = [(size, 0), (-size, -size/2), (-size, size/2)]
        
        vertices = []
        for dy, dx in local_coords:
            # Apply 2D rotation matrix
            r_lat = lat + (dy * cos_b - dx * sin_b)
            r_lon = lon + (dy * sin_b + dx * cos_b) * lon_scale
            vertices.append([r_lat, r_lon])
            
        return vertices

    def plot_trajectory_folium(self, arrow_step: int = 10, arrow_size: float = 0.002) -> folium.Map:
        """
        Plots the vessel trajectory on an interactive Folium map.
        - The track is color-graded by actual time.
        - Adds directional arrows for clarity when zoomed in.
        - Clusters overlapping Arrival/Departure markers and numbers them chronologically.
        """
        from folium import plugins
        if self.clean_df is None or 'LATITUDE(DD)' not in self.clean_df.columns:
            raise ValueError("Coordinates not processed. Call process_coordinates() first.")

        # 1. Filter valid GPS data
        geo_data = self.clean_df.replace(
            {'LATITUDE(DD)': 0, 'LONGITUDE(DD)': 0}, np.nan
        ).dropna(subset=['LATITUDE(DD)', 'LONGITUDE(DD)'])
        
        if len(geo_data) == 0:
            raise ValueError("No valid coordinates found.")

        coords = geo_data[['LATITUDE(DD)', 'LONGITUDE(DD)']].values.tolist()
        
        # 2. Time-based Color Grading using Elapsed Hours
        # Calculate elapsed time in hours relative to the first valid GPS point
        start_time = geo_data.index.min()
        elapsed_seconds = (geo_data.index - start_time).total_seconds()
        time_proxy = elapsed_seconds / 3600.0  # Convert to hours
        
        vmin, vmax = time_proxy.min(), time_proxy.max()
        
        colormap = cm.LinearColormap(
            colors=['blue', 'cyan', 'yellow', 'red'], 
            vmin=vmin, 
            vmax=vmax,
            caption=f'Elapsed Time (Hours) since {start_time.strftime("%d/%m/%Y %H:%M")}'
        )

        # 3. Initialize Map
        center_coord = [geo_data['LATITUDE(DD)'].median(), geo_data['LONGITUDE(DD)'].median()]
        vessel_map = folium.Map(location=center_coord, zoom_start=6, tiles='CartoDB positron')

        # 4. Draw the graded trajectory line
        folium.ColorLine(
            positions=coords,
            colors=time_proxy,
            colormap=colormap,
            weight=4,
            opacity=0.7
        ).add_to(vessel_map)

        # 5. Add Dynamic Arrows
        for i in range(0, len(geo_data) - 1, arrow_step):
            p1 = geo_data.iloc[i]
            p2 = geo_data.iloc[i+1]
            
            # Calculate bearing and color
            bearing = self._calculate_bearing(p1['LATITUDE(DD)'], p1['LONGITUDE(DD)'], 
                                              p2['LATITUDE(DD)'], p2['LONGITUDE(DD)'])
            
            # Get hex color from the colormap for this specific time
            color_hex = colormap(time_proxy[i])
            
            # Generate vertices
            arrow_verts = self._get_arrow_vertices(p1['LATITUDE(DD)'], p1['LONGITUDE(DD)'], bearing, size=arrow_size)
            
            folium.Polygon(
                locations=arrow_verts,
                color=color_hex,
                fill=True,
                fill_color=color_hex,
                fill_opacity=0.9,
                weight=1,
                popup=f"Time: {geo_data.index[i].strftime('%H:%M')}"
            ).add_to(vessel_map)
        
        colormap.add_to(vessel_map)

        # 6. Detect & Cluster Arrivals and Departures
        if 'STATUS' in geo_data.columns:
            port_states = ['idle', 'loading', 'discharging']
            sea_states = ['laden', 'ballast']

            status_lower = geo_data['STATUS'].str.lower()
            state_map = pd.Series(np.nan, index=geo_data.index)
            state_map[status_lower.isin(port_states)] = 0
            state_map[status_lower.isin(sea_states)] = 1
            state_map = state_map.ffill()

            transitions = state_map.diff()
            departures = geo_data[transitions == 1]
            arrivals = geo_data[transitions == -1]

            event_cluster = plugins.MarkerCluster(name="Port Events").add_to(vessel_map)

            for i, (timestamp, row) in enumerate(departures.iterrows(), start=1):
                popup_html = f"<b>Departure #{i}</b><br>Time: {timestamp}<br>Status: {row['STATUS']}"
                folium.Marker(
                    location=[row['LATITUDE(DD)'], row['LONGITUDE(DD)']],
                    popup=folium.Popup(popup_html, max_width=250),
                    icon=folium.Icon(color='green', icon='arrow-up', prefix='fa')
                ).add_to(event_cluster)

            for i, (timestamp, row) in enumerate(arrivals.iterrows(), start=1):
                popup_html = f"<b>Arrival #{i}</b><br>Time: {timestamp}<br>Status: {row['STATUS']}"
                folium.Marker(
                    location=[row['LATITUDE(DD)'], row['LONGITUDE(DD)']],
                    popup=folium.Popup(popup_html, max_width=250),
                    icon=folium.Icon(color='red', icon='arrow-down', prefix='fa')
                ).add_to(event_cluster)

        return vessel_map