import xarray as xr
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf
import warnings
warnings.filterwarnings('ignore')

class WeatherCalibrator:
    def __init__(self, nc_path_x, nc_path_y, ports_dict, schedule_list):
        """Initializes the calibrator and dynamically builds the node topology."""
        print("Loading NetCDF datasets...")
        self.ds_x = xr.open_dataset(nc_path_x)
        self.ds_y = xr.open_dataset(nc_path_y)
        
        self.var_x = list(self.ds_x.data_vars)[0]
        self.var_y = list(self.ds_y.data_vars)[0]
        
        self.lat_key = 'latitude' if 'latitude' in self.ds_x.coords else 'lat'
        self.lon_key = 'longitude' if 'longitude' in self.ds_x.coords else 'lon'

        # Build dynamic nodes (Ports, Transits, and Global Ref)
        self.nodes = self._build_topology(ports_dict, schedule_list)
        
        # Pre-calculate valid sea points to avoid NaNs (Land Masking)
        self.valid_sea_points = self._get_valid_sea_mask()

    def _build_topology(self, ports, schedule):
        """Builds a dictionary of all relevant Lat/Lon points."""
        nodes = {}
        
        # 1. Add Ports
        for port, coords in ports.items():
            nodes[port] = (coords['lat'], coords['lon'])
            
        # 2. Add Transit Midpoints based on the schedule
        for leg in schedule:
            start, end = leg['start'], leg['end']
            transit_name = f"Transit_{start}_{end}"
            
            # Simple geographic midpoint (sufficient for short Aegean distances)
            mid_lat = (ports[start]['lat'] + ports[end]['lat']) / 2.0
            mid_lon = (ports[start]['lon'] + ports[end]['lon']) / 2.0
            nodes[transit_name] = (mid_lat, mid_lon)
            
        # 3. Add Global Reference Node (Average of all ports)
        avg_lat = np.mean([c['lat'] for c in ports.values()])
        avg_lon = np.mean([c['lon'] for c in ports.values()])
        nodes['Global_Ref'] = (avg_lat, avg_lon)
        
        return nodes

    def _get_valid_sea_mask(self):
        """Extracts a DataFrame of all grid coordinates that contain valid data (not NaN)."""
        # Take the first time slice of the X-stress data
        sample_slice = self.ds_x[self.var_x].isel(time=0)
        # Convert to dataframe and drop NaNs. What remains are strictly valid sea points.
        valid_coords = sample_slice.to_dataframe().dropna().reset_index()
        return valid_coords[[self.lat_key, self.lon_key]]

    def _get_nearest_sea_point(self, target_lat, target_lon):
        """Finds the closest marine grid cell, guaranteeing no NaNs."""
        df_valid = self.valid_sea_points.copy()
        
        # Calculate squared Euclidean distance (fast and effective for finding nearest)
        df_valid['dist'] = (df_valid[self.lat_key] - target_lat)**2 + (df_valid[self.lon_key] - target_lon)**2
        
        # Get the lat/lon of the minimum distance
        nearest = df_valid.loc[df_valid['dist'].idxmin()]
        return nearest[self.lat_key], nearest[self.lon_key]

    def _extract_and_compute_magnitude(self):
        """Extracts valid marine data for all nodes and computes scalar magnitude."""
        df_list = []
        for node_name, (lat, lon) in self.nodes.items():
            # 1. Snap to the nearest actual SEA point
            sea_lat, sea_lon = self._get_nearest_sea_point(lat, lon)
            
            # 2. Extract data safely
            node_x = self.ds_x.sel({self.lat_key: sea_lat, self.lon_key: sea_lon}, method='nearest')
            node_y = self.ds_y.sel({self.lat_key: sea_lat, self.lon_key: sea_lon}, method='nearest')
            
            tau_x = node_x[self.var_x].values
            tau_y = node_y[self.var_y].values
            magnitude = np.sqrt(tau_x**2 + tau_y**2)
            
            df = pd.DataFrame({
                'time': node_x['time'].values,
                'node': node_name,
                'magnitude': magnitude
            })
            df_list.append(df)
            
        return pd.concat(df_list, ignore_index=True)

    def process_and_normalize(self):
        df = self._extract_and_compute_magnitude()
        global_min, global_max = df['magnitude'].min(), df['magnitude'].max()
        df['W'] = (df['magnitude'] - global_min) / (global_max - global_min)
        df['year'], df['month'] = df['time'].dt.year, df['time'].dt.month
        return df

    def _calculate_theta(self, series, dt_hours=24):
        if len(series) < 2: return np.nan
        acf_vals = acf(series, nlags=min(30, len(series)-1), fft=True)
        threshold = np.exp(-1)
        below_thresh = np.where(acf_vals < threshold)[0]
        tau_lag = below_thresh[0] * dt_hours if len(below_thresh) > 0 else len(acf_vals) * dt_hours 
        return 1.0 / tau_lag if tau_lag > 0 else np.nan

    def calibrate_month(self, df, target_month):
        df_month = df[df['month'] == target_month].dropna()
        df_ref = df_month[df_month['node'] == 'Global_Ref']
        
        yearly_params = []
        for year, group in df_ref.groupby('year'):
            W_y = group['W'].values
            if len(W_y) < 2: continue
            mu_y, var_y = np.mean(W_y), np.var(W_y)
            theta_y = self._calculate_theta(W_y, dt_hours=24) 
            if 0 < mu_y < 1 and theta_y > 0:
                sigma_y = np.sqrt(max(0, (2 * theta_y * var_y) / (mu_y * (1 - mu_y))))
                yearly_params.append({'mu': mu_y, 'theta': theta_y, 'sigma': sigma_y})
                
        global_params = pd.DataFrame(yearly_params).mean().to_dict()
        
        k_factors = {}
        for node in self.nodes.keys():
            if node == 'Global_Ref': continue
            mu_i = df_month[df_month['node'] == node]['W'].mean()
            k_factors[node] = mu_i / global_params['mu']
            
        return global_params, k_factors