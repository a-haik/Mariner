import matplotlib.pyplot as plt
import folium
import pandas as pd
import numpy as np
import math
from typing import List, Optional
import branca.colormap as cm
import matplotlib.patches as mpatches

class VesselVisualizer:
    """Handles all plotting and map generation for vessel telemetry."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
        # Pragmatic, physically-grouped color palette
        self.status_colors = {
            # 1. Transit (Cool colors)
            'sea going': '#4C72B0',   # Standard Blue
            'laden': '#55A868',       # Sea Green
            'ballast': '#81D8D0',     # Light Teal
            
            # 2. Tank Operations at Sea (Warm/Purple colors)
            'tank cleaning': '#C44E52', # Muted Red
            'tank heating': '#8172B3',  # Muted Purple
            'gas freeing': '#CCB974',   # Khaki/Yellowish
            
            # 3. Cargo Operations (Orange/Brown colors)
            'loading': '#DD8452',       # Orange
            'unloading': '#D65F5F',     # Coral/Light Orange
            'discharging': '#D65F5F',   # Same as unloading
            
            # 4. Maneuvering and Idle (Red for high stress, Grey for baseline)
            'at port in/out': '#8C2D04', # Dark, intense Red (Highest Load)
            'at harbour': '#C4C4C4',     # Light Grey
            'idle': '#EAEAEA'            # Very Light Grey
        }

    def plot_series(self, 
                    columns: List[str], 
                    subplots: bool = False,
                    rolling_window: Optional[str] = None, 
                    secondary_y: Optional[str] = None,
                    status_col: str = 'STATUS',
                    predicted_col: Optional[str] = None, # NEW: Pass the HMM column name here
                    **kwargs) -> tuple:
        """A flexible plotting tool for time-series data with HMM overlay."""
        
        if self.df is None or self.df.empty:
            raise ValueError("Dataframe is empty or not loaded.")

        # If we have a predicted column, we add one extra row for the step plot
        n_signal_rows = len(columns) if subplots else 1
        total_rows = n_signal_rows + (1 if predicted_col else 0)
        
        # Adjust height ratio so the discrete step plot is smaller
        gridspec_kw = {'height_ratios': [3]*n_signal_rows + [1]} if predicted_col else None
        
        kwargs.setdefault('figsize', (14, 4 * total_rows))
        fig, axes = plt.subplots(nrows=total_rows, ncols=1, gridspec_kw=gridspec_kw, **kwargs)
        
        if not isinstance(axes, (list, np.ndarray)):
            axes = [axes]

        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        
        show_status = status_col in self.df.columns
        if show_status:
            status_changes = (self.df[status_col] != self.df[status_col].shift()).fillna(True)
            change_indices = self.df.index[status_changes].tolist()
            change_indices.append(self.df.index[-1])

        all_handles = []
        all_labels = []

        # 1. Plot Continuous Signals
        for i, col in enumerate(columns):
            if col not in self.df.columns:
                continue
                
            ax = axes[i] if subplots else axes[0]
            is_secondary = (not subplots and col == secondary_y)
            plot_ax = ax.twinx() if is_secondary else ax
            color = colors[i % len(colors)]
            
            line, = plot_ax.plot(self.df.index, self.df[col], alpha=0.8, color=color, label=f'{col} (Raw)')
            all_handles.append(line)
            all_labels.append(f'{col} (Raw)')
            
            if rolling_window:
                smoothed = self.df[col].rolling(window=rolling_window, center=True).mean()
                s_line, = plot_ax.plot(self.df.index, smoothed, linewidth=2, color=color, 
                                       linestyle='--', label=f'{col} (Mean {rolling_window})')
                all_handles.append(s_line)
                all_labels.append(f'{col} (Mean {rolling_window})')

            # Add Background Color Blocks (Human Status)
            if show_status and (subplots or i == 0):
                for start, end in zip(change_indices[:-1], change_indices[1:]):
                    current_status = self.df.loc[start, status_col]
                    bg_color = self.status_colors.get(str(current_status).lower(), 'white')
                    ax.axvspan(start, end, color=bg_color, alpha=0.3, zorder=-1)

            plot_ax.set_ylabel(col, color=color if is_secondary else 'black')

        # 2. Plot Discrete HMM Predictions (NEW)
        if predicted_col and predicted_col in self.df.columns:
            pred_ax = axes[-1]
            pred_ax.step(self.df.index, self.df[predicted_col], where='post', color='black', linewidth=2)
            pred_ax.set_ylabel("HMM State", fontweight='bold')
            pred_ax.set_yticks(np.unique(self.df[predicted_col].dropna()))
            pred_ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Generate Unified Legends
        if subplots:
            for ax in axes[:n_signal_rows]:
                ax.legend(loc='upper left')
        else:
            axes[0].legend(all_handles, all_labels, loc='upper left')

        if show_status:
            unique_statuses = self.df[status_col].dropna().astype(str).str.lower().unique()
            patches = [mpatches.Patch(color=color, label=status.title(), alpha=0.3) 
                       for status, color in self.status_colors.items() 
                       if status in unique_statuses]
            fig.legend(handles=patches, title=f"Vessel Status ({status_col})", 
                       loc='center left', bbox_to_anchor=(1, 0.5))

        axes[-1].set_xlabel("Time")
        fig.suptitle(f"MARINER Telemetry vs. Unsupervised HMM ({predicted_col})", y=1.02, fontweight='bold')
        fig.tight_layout()
            
        return fig, axes

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

    def plot_trajectory_folium(self, status_col: str = 'STATUS', arrow_step: int = 10, arrow_size: float = 0.002) -> folium.Map:
        """Plots the vessel trajectory on an interactive Folium map."""
        from folium import plugins
        if self.df is None or 'LATITUDE(DD)' not in self.df.columns:
            raise ValueError("Coordinates not processed.")

        geo_data = self.df.replace({'LATITUDE(DD)': 0, 'LONGITUDE(DD)': 0}, np.nan).dropna(subset=['LATITUDE(DD)', 'LONGITUDE(DD)'])
        if len(geo_data) == 0:
            raise ValueError("No valid coordinates found.")

        coords = geo_data[['LATITUDE(DD)', 'LONGITUDE(DD)']].values.tolist()
        
        start_time = geo_data.index.min()
        elapsed_seconds = (geo_data.index - start_time).total_seconds()
        time_proxy = elapsed_seconds / 86400.0  # Convert to days
        
        colormap = cm.LinearColormap(
            colors=['blue', 'cyan', 'yellow', 'red'], 
            vmin=time_proxy.min(), vmax=time_proxy.max(),
            caption=f'Elapsed Time (Days) since {start_time.strftime("%d/%m/%Y %H:%M")}'
        )

        center_coord = [geo_data['LATITUDE(DD)'].median(), geo_data['LONGITUDE(DD)'].median()]
        vessel_map = folium.Map(location=center_coord, zoom_start=6, tiles='CartoDB positron')

        folium.ColorLine(positions=coords, colors=time_proxy, colormap=colormap, weight=4, opacity=0.7).add_to(vessel_map)

        for i in range(0, len(geo_data) - 1, arrow_step):
            p1, p2 = geo_data.iloc[i], geo_data.iloc[i+1]
            bearing = self._calculate_bearing(p1['LATITUDE(DD)'], p1['LONGITUDE(DD)'], p2['LATITUDE(DD)'], p2['LONGITUDE(DD)'])
            arrow_verts = self._get_arrow_vertices(p1['LATITUDE(DD)'], p1['LONGITUDE(DD)'], bearing, size=arrow_size)
            
            folium.Polygon(
                locations=arrow_verts, color=colormap(time_proxy[i]), fill=True,
                fill_color=colormap(time_proxy[i]), fill_opacity=0.9, weight=1,
                popup=f"Time: {geo_data.index[i].strftime('%H:%M')}"
            ).add_to(vessel_map)
        
        colormap.add_to(vessel_map)

        # Detect & Cluster Arrivals and Departures using the dynamic status column
        if status_col in geo_data.columns:
            # Added new regimes to the port/sea mapping logic
            port_states = ['idle', 'loading', 'discharging', 'at harbour', 'unloading', 'at port in/out']
            sea_states = ['laden', 'ballast', 'sea going', 'tank cleaning', 'tank heating', 'gas freeing']

            status_lower = geo_data[status_col].str.lower()
            state_map = pd.Series(np.nan, index=geo_data.index)
            state_map[status_lower.isin(port_states)] = 0
            state_map[status_lower.isin(sea_states)] = 1
            state_map = state_map.ffill()

            transitions = state_map.diff()
            departures = geo_data[transitions == 1]
            arrivals = geo_data[transitions == -1]

            event_cluster = plugins.MarkerCluster(name="Port Events").add_to(vessel_map)

            for i, (timestamp, row) in enumerate(departures.iterrows(), start=1):
                popup_html = f"<b>Departure #{i}</b><br>Time: {timestamp}<br>Regime: {row[status_col]}"
                folium.Marker(location=[row['LATITUDE(DD)'], row['LONGITUDE(DD)']], popup=folium.Popup(popup_html, max_width=250), icon=folium.Icon(color='green', icon='arrow-up', prefix='fa')).add_to(event_cluster)

            for i, (timestamp, row) in enumerate(arrivals.iterrows(), start=1):
                popup_html = f"<b>Arrival #{i}</b><br>Time: {timestamp}<br>Regime: {row[status_col]}"
                folium.Marker(location=[row['LATITUDE(DD)'], row['LONGITUDE(DD)']], popup=folium.Popup(popup_html, max_width=250), icon=folium.Icon(color='red', icon='arrow-down', prefix='fa')).add_to(event_cluster)

        return vessel_map