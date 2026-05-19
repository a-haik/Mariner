# src/visualizer.py
import matplotlib.pyplot as plt
import folium
import pandas as pd
import numpy as np
import math
from typing import List, Optional
import branca.colormap as cm
import matplotlib.patches as mpatches
import plotly.graph_objects as go
from src.config import COLOR, PHYSICS

class VesselVisualizer:
    """Handles all plotting and map generation for vessel telemetry."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
        # Pragmatic, physically-grouped color palette
        self.status_colors = COLOR.status_colors

    def plot_series(self, 
                    columns: List[str], 
                    subplots: bool = False,
                    rolling_window: Optional[str] = None, 
                    secondary_y: Optional[str] = None,
                    status_col: str = 'STATUS',

                    **kwargs) -> tuple:
        """A flexible plotting tool for time-series"""
        
        if self.df is None or self.df.empty:
            raise ValueError("Dataframe is empty or not loaded.")

        n_signal_rows = len(columns) if subplots else 1
        
        kwargs.setdefault('figsize', (14, 4 * n_signal_rows))
        fig, axes = plt.subplots(nrows=n_signal_rows, ncols=1, **kwargs)
        
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
        fig.suptitle("MARINER Telemetry", y=1.02, fontweight='bold')
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
        """Plots the vessel trajectory on an interactive Folium map with true Arrival/Departure markers."""
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

        # Detect true Arrivals and Departures
        if status_col in geo_data.columns:
            # Robust mapping that covers both the raw logs and our new engineered phases
            port_states = [
                'idle', 'loading', 'discharging', 'unloading',
                'port_idle', 'port_loading', 'port_unloading',
            ]
            
            status_lower = geo_data[status_col].astype(str).str.lower()
            
            # Create a boolean series: True if at port, False if at sea
            is_at_port = status_lower.isin(port_states)
            
            # Shift the boolean series to detect the exact moment of state change
            # True -> False = Departure (Port to Sea)
            # False -> True = Arrival (Sea to Port)
            port_transitions = is_at_port.astype(int).diff()
            
            departures = geo_data[port_transitions == -1]
            arrivals = geo_data[port_transitions == 1]

            event_cluster = plugins.MarkerCluster(name="Port Events").add_to(vessel_map)

            for i, (timestamp, row) in enumerate(departures.iterrows(), start=1):
                popup_html = f"<b>Departure #{i}</b><br>Time: {timestamp}<br>Status: {row[status_col]}"
                folium.Marker(
                    location=[row['LATITUDE(DD)'], row['LONGITUDE(DD)']], 
                    popup=folium.Popup(popup_html, max_width=250), 
                    icon=folium.Icon(color='green', icon='arrow-up', prefix='fa')
                ).add_to(event_cluster)

            for i, (timestamp, row) in enumerate(arrivals.iterrows(), start=1):
                popup_html = f"<b>Arrival #{i}</b><br>Time: {timestamp}<br>Status: {row[status_col]}"
                folium.Marker(
                    location=[row['LATITUDE(DD)'], row['LONGITUDE(DD)']], 
                    popup=folium.Popup(popup_html, max_width=250), 
                    icon=folium.Icon(color='red', icon='arrow-down', prefix='fa')
                ).add_to(event_cluster)

        return vessel_map
    
    def plot_brick_space(self, registry_df: pd.DataFrame, y_axis_metric: str = 'Normalized_Fatigue_Rate', include_loitering: bool = True) -> go.Figure:
        """
        Constructs an interactive Plotly scatter plot for individual physical blocks.
        """
        if registry_df.empty:
            raise ValueError("Input registry dataframe is completely empty.")
            
        plot_df = registry_df.copy()
        fig = go.Figure()
        
        # Calculate consistent sizing reference based on the maximum duration
        max_duration = plot_df['Duration_h'].max()
        sizeref_val = 2. * max_duration / (40.**2) # Bounds max bubble size to ~40px
        
        unique_phases = plot_df['PHASE'].unique()
        
        # 1. Plot the True Data Points
        for phase in unique_phases:
            phase_df = plot_df[plot_df['PHASE'] == phase]
            color = self.status_colors.get(phase, '#000000')
            
            # Calculate midpoint and symmetric error
            x_mid = (phase_df['H2_Rate_Lower_kg_h'] + phase_df['H2_Rate_Upper_kg_h']) / 2.0
            err_val = phase_df['H2_Rate_Upper_kg_h'] - x_mid
            
            hover_text = []
            for _, row in phase_df.iterrows():
                is_transit = row['PHASE'] in ['Sea_Transit_Laden', 'Sea_Transit_Ballast']
                handling_info = f"With Loitering: {row.get('With_Loitering')}<br>" if is_transit else ""
                
                hover_str = (
                    f"<b>Phase Block: {row['PHASE']}</b><br>"
                    f"File Origin: {row['Source_File']}<br>"
                    f"{handling_info}"
                    f"Start Timestamp: {row['Start_Time']}<br>"
                    f"Total Duration: {row['Duration_h']:.2f} h<br>"
                    f"Mean Power Demand: {row['Mean_Power_kW']:.1f} kW<br>"
                    f"-----------------------------------<br>"
                    f"H2 Rate: [{row['H2_Rate_Lower_kg_h']:.2f} - {row['H2_Rate_Upper_kg_h']:.2f}] kg/h<br>"
                    f"Fatigue Damage Rate: {row.get('Fatigue_Damage_Rate', 0.0):.4f} (/s)<br>"
                    f"Power Fluctuation Intensity: {row.get('Mean_Power_Fluctuation_Intensity', 0.0):.2f}(kW^2/s^2)<br>"

                )
                hover_text.append(hover_str)

            fig.add_trace(go.Scatter(
                x=x_mid,
                y=phase_df[y_axis_metric],
                mode='markers', # Strict markers, no text labels
                name=phase,
                text=hover_text,
                hoverinfo='text',
                legendgroup="Phases",
                legendgrouptitle_text="Operational Regimes",
                marker=dict(
                    size=phase_df['Duration_h'],
                    sizemode='area',
                    sizeref=sizeref_val,
                    sizemin=4,
                    color=color,
                    line=dict(width=1, color='DarkSlateGrey')
                ),
                error_x=dict(
                    type='data',
                    symmetric=True,
                    array=err_val,
                    color='rgba(120,120,120,0.35)',
                    thickness=1.5,
                    width=4
                )
            ))
            
        # 2. Inject Dummy Traces for the Size Legend
        legend_sizes_h = [
            round(plot_df['Duration_h'].quantile(0.1)),
            round(plot_df['Duration_h'].median()), 
            round(max_duration)
        ]
        
        for s in legend_sizes_h:
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                name=f"{s} hours",
                legendgroup="Size",
                legendgrouptitle_text="Block Duration",
                marker=dict(
                    size=[s],
                    sizemode='area',
                    sizeref=sizeref_val,
                    color='rgba(150,150,150,0.5)',
                    line=dict(width=1, color='DarkSlateGrey')
                )
            ))
            
        fig.update_layout(
            title=dict(
                text=f"MARINER Technical Brick Workspace Axis Profile ({y_axis_metric})",
                font=dict(size=14, family="Arial", color="black")
            ),
            xaxis_title=rf"Expected Hydrogen Flow Rate (kg/h) [Error Bars: η ∈ [{PHYSICS.ETA_LOWER:.2f}, {PHYSICS.ETA_UPPER:.2f}]]",
            yaxis_title=f"PEMFC Degradation Index: {y_axis_metric}",
            template="plotly_white",
            hovermode='closest',
            width=1100,
            height=650,
        )
        
        return fig


    def plot_phase_statistics(self, stats_df: pd.DataFrame, y_axis_metric: str = 'Normalized_Fatigue_Rate') -> go.Figure:
        """
        Plots aggregated expected values for global operational states.
        Bubble size represents the fractional time spent in that state.
        """
        if stats_df.empty:
            raise ValueError("Statistics DataFrame is empty.")
            
        total_time = stats_df['Total_Logged_Hours'].sum()
        fig = go.Figure()

        sizeref_val = 2. * max(stats_df['Total_Logged_Hours'] / total_time) / (50.**2)

        for phase, row in stats_df.iterrows():
            if phase == 'Unknown': 
                continue
                
            color = self.status_colors.get(phase, '#333333')
            time_fraction = row['Total_Logged_Hours'] / total_time
            
            h2_lower = row['Mean_H2_Rate_Lower_kg_h']
            h2_upper = row['Mean_H2_Rate_Upper_kg_h']
            h2_mid = (h2_lower + h2_upper) / 2.0
            err_val = h2_upper - h2_mid

            hover_text = (
                f"<b>{phase}</b><br>"
                f"Time Fraction: {time_fraction*100:.1f}% ({row['Total_Logged_Hours']:.1f} hrs)<br>"
                f"Mean Power: {row['Mean_Power_kW']:.1f} kW<br>"
                f"H2 Rate: [{h2_lower:.2f} - {h2_upper:.2f}] kg/h<br>"
                f"Fatigue Damage Rate: {row['Fatigue_Damage_Rate']:.4f} (/s)<br>"
                f"Power Fluctuation Intensity: {row['Mean_Power_Fluctuation_Intensity']:.2f}(kW^2/s^2)<br>"
            )

            fig.add_trace(go.Scatter(
                x=[h2_mid],
                y=[row['Mean_Power_Fluctuation_Intensity']],
                mode='markers', # Stripped text labels to maintain clean layout
                name=phase,
                hoverinfo='text',
                hovertext=[hover_text],
                legendgroup="Phases",
                legendgrouptitle_text="Aggregated Regimes",
                marker=dict(
                    size=[time_fraction],
                    sizemode='area',
                    sizeref=sizeref_val,
                    sizemin=10,
                    color=color,
                    line=dict(width=1.5, color='DarkSlateGrey')
                ),
                error_x=dict(
                    type='data', array=[err_val], arrayminus=[err_val],
                    color='rgba(100,100,100,0.5)', thickness=2, width=5
                )
            ))

        # Dummy Traces for Time Fraction Size Legend
        legend_fractions = [0.1, 0.5, 1.0] # 10%, 50%, 100%
        for f in legend_fractions:
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                name=f"{int(f*100)}% of Total Time",
                legendgroup="Size",
                legendgrouptitle_text="Time Fraction",
                marker=dict(
                    size=[f],
                    sizemode='area',
                    sizeref=sizeref_val,
                    color='rgba(150,150,150,0.5)',
                    line=dict(width=1, color='DarkSlateGrey')
                )
            ))

        fig.update_layout(
            title="Global Phase Statistics: H2 Flow vs PEMFC Degradation",
            xaxis_title=rf"Expected Hydrogen Flow Rate (kg/h) [Error Bars: η ∈ [{PHYSICS.ETA_LOWER:.2f}, {PHYSICS.ETA_UPPER:.2f}]]",
            yaxis_title=f"PEMFC Degradation Index: {y_axis_metric}",
            template="plotly_white",
            hovermode='closest',
            width=1100,
            height=650
        )
        
        return fig