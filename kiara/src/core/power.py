import numpy as np
import pandas as pd
from config.vessel_specs import POWER_CONFIG, STATE_PHYSICS, CLIMATE_STATS

class PowerGenerator:
    def __init__(self, config):
        self.config = config

    def _apply_first_order_inertia(self, p_commanded, tau, dt=1.0):
        """
        Applies a first-order low-pass filter to simulate physical engine inertia
        and governor response lag.
        """
        num_steps = len(p_commanded)
        p_actual = np.zeros(num_steps)
        p_actual[0] = p_commanded[0]
        
        # Calculate the discrete filter coefficient
        alpha = dt / max(dt, tau)
        
        for t in range(1, num_steps):
            p_actual[t] = p_actual[t-1] + alpha * (p_commanded[t] - p_actual[t-1])
            
        return p_actual

    def _generate_dynamic_ou_noise(self, num_seconds, p_commanded, theta_t, sigma_inf_t):
        dt = 1.0
        chi = np.zeros(num_seconds)
        dW = np.random.normal(0, np.sqrt(dt), num_seconds)
        
        for t in range(1, num_seconds):
            drift = -theta_t[t-1] * chi[t-1] * dt
            
            # Calibrate diffusion using the specific command's target capacity envelope
            sigma_SDE = (p_commanded[t-1] * sigma_inf_t[t-1]) * np.sqrt(2.0 * theta_t[t-1])
            
            chi[t] = chi[t-1] + drift + (sigma_SDE * dW[t])
        return chi

    def _build_mode_dependent_parameters(self, df_timeline, total_seconds, base_sigma):
        theta_t = np.zeros(total_seconds)
        sigma_inf_t = np.zeros(total_seconds)
        
        current_sec = 0
        for _, row in df_timeline.iterrows():
            duration_secs = int((row['end_time'] - row['start_time']).total_seconds())
            state = row['state']
            
            physics = STATE_PHYSICS.get(state, STATE_PHYSICS['transit'])
            end_sec = min(current_sec + duration_secs, total_seconds)
            
            theta_t[current_sec:end_sec] = 1.0 / max(1.0, physics['tau'])
            sigma_inf_t[current_sec:end_sec] = base_sigma * physics['sigma_multiplier']
            
            current_sec = end_sec
            
        return theta_t, sigma_inf_t

    def generate_traces(self, df_timeline, weather_global, k_factors, tunable_params, month_mu, dt_seconds=1):
        total_seconds = int((df_timeline['end_time'].iloc[-1] - df_timeline['start_time'].iloc[0]).total_seconds())
        
        # 1. Initialize second-by-second target arrays
        p_main_commanded = np.zeros(total_seconds)
        p_hotel_commanded = np.zeros(total_seconds)
        p_thruster_commanded = np.zeros(total_seconds)
        state_series = []
        
        # Pre-calculate climate boundaries for thrusters
        w_mean = CLIMATE_STATS['W_annual_mean']
        w_std = CLIMATE_STATS['W_annual_std']
        w_cut_in = w_mean + self.config['alpha_thruster_start'] * w_std
        w_saturation = w_mean + self.config['beta_thruster_max'] * w_std
        
        # Build second-by-second baseline curves from state nomenclature
        current_sec = 0
        for _, row in df_timeline.iterrows():
            duration_secs = int((row['end_time'] - row['start_time']).total_seconds())
            state = row['state']
            
            for _ in range(duration_secs):
                if current_sec >= total_seconds: break
                w_curr = weather_global[current_sec]
                state_series.append(state)
                
                # Dynamic weather factors used across multiple states
                w_factor = np.clip((w_curr - w_mean) / (w_std * 2.0), 0.0, 1.0)
                
                # Calculate active thruster scaling factor if weather breaks the threshold
                if w_curr > w_cut_in:
                    t_factor = np.clip((w_curr - w_cut_in) / (w_saturation - w_cut_in), 0.0, 1.0)
                else:
                    t_factor = 0.0

                # --- MODE DISPATCHING LOGIC ---
                if state == 'transit':
                    wave_res_mult = 1.0 + (w_curr * tunable_params.get('wave_resistance_factor', 0.0))
                    p_main_commanded[current_sec] = self.config['P_main_transit'] * wave_res_mult
                    
                    # Transit auxiliary load is fixed at 210 kW out at sea
                    p_hotel_commanded[current_sec] = self.config['P_aux_transit']
                    p_thruster_commanded[current_sec] = 0.0
                    
                elif state == 'maneuvering':
                    # Main propulsion scales dynamically to fight wind forces (5000 kW to 5400 kW)
                    p_main_commanded[current_sec] = self.config['P_main_maneuver_base'] + w_factor * (self.config['P_main_maneuver_max'] - self.config['P_main_maneuver_base'])
                    
                    # Combined electrical infrastructure (Capped at 350 kW total load)
                    p_hotel_commanded[current_sec] = self.config['P_aux_maneuver_base']
                    p_thruster_commanded[current_sec] = t_factor * (self.config['P_aux_thruster_max'] - self.config['P_aux_maneuver_base'])
                        
                elif state == 'port_operations':
                    # NEW: Main propulsion climbs up to 5400 kW under severe weather conditions to help pin the hull
                    # At zero wind scaling (w_factor=0), it stays at baseline 5000 kW. 
                    # At max wind scaling (w_factor=1), it scales up to 5400 kW.
                    p_main_base_ops = self.config['P_main_port_ops']  # 5000 kW
                    p_main_max_ops = self.config['P_main_maneuver_max'] # 5400 kW
                    
                    p_main_commanded[current_sec] = p_main_base_ops + w_factor * (p_main_max_ops - p_main_base_ops)
                    
                    # Thrusters remain operationally available to counter wind gusts while moored (Capped at 350 kW)
                    p_hotel_commanded[current_sec] = self.config['P_aux_port_ops']
                    p_thruster_commanded[current_sec] = t_factor * (self.config['P_aux_thruster_max'] - self.config['P_aux_port_ops'])
                    
                elif state == 'idling':
                    # Secure main propulsion entirely (cold ironed / overnight)
                    p_main_commanded[current_sec] = self.config['P_main_idling']
                    
                    # pure structural hotel baseline load, no thrusters active
                    p_hotel_commanded[current_sec] = self.config['P_aux_idling']
                    p_thruster_commanded[current_sec] = 0.0
                    
                current_sec += 1

        # 2. Execute Dynamic Non-Stationary SDE calculations
        base_sigma = tunable_params.get('sigma_fraction', 0.03)
        theta_t, sigma_inf_t = self._build_mode_dependent_parameters(df_timeline, total_seconds, base_sigma)
        
        # Calculate decoupled noise tracks using clear, un-scaled target envelopes
        ou_noise_main = self._generate_dynamic_ou_noise(total_seconds, p_main_commanded, theta_t, sigma_inf_t)
        
        # Scale the hotel load standard deviation envelope downward by 75%
        sigma_inf_hotel = sigma_inf_t * self.config['aux_volatility_reduction']
        ou_noise_hotel = self._generate_dynamic_ou_noise(total_seconds, p_hotel_commanded, theta_t, sigma_inf_hotel)
        
        ou_noise_thruster = self._generate_dynamic_ou_noise(total_seconds, p_thruster_commanded, theta_t, sigma_inf_t)
        
        # A. Compile raw noisy commands (Commanded + OU Environmental Noise)
        p_main_noisy = np.maximum(0.0, p_main_commanded + ou_noise_main)
        p_aux_noisy = np.maximum(0.0, p_hotel_commanded + p_thruster_commanded + ou_noise_hotel + ou_noise_thruster)
        
        # B. Apply Physical Inertia Filter (NEW)
        # Main engines feel the heavy 15-second diesel/thermal lag
        p_main_physical = self._apply_first_order_inertia(p_main_noisy, self.config['tau_diesel'], dt=1.0)
        
        # Auxiliaries/Thrusters feel the responsive 1-second electric motor lag
        p_aux_physical = self._apply_first_order_inertia(p_aux_noisy, self.config['tau_electric'], dt=1.0)
        
        # SCALE SENSOR FUZZ: Map the slider percentage so that the total peak-to-peak 
        # bounds match the user's intent, treating the slider value as the 3-sigma maximum envelope limit.
        max_fuzz_percentage = tunable_params.get('delta_instrument', 0.005)
        sigma_sensor_fuzz = max_fuzz_percentage / 3.0
        
        # Inject High-Frequency Telemetry Sensor Error using calibrated standard deviation
        fuzz_main = np.random.normal(0, sigma_sensor_fuzz * p_main_physical)
        fuzz_aux = np.random.normal(0, sigma_sensor_fuzz * p_aux_physical)
        
        timestamps = pd.date_range(start=df_timeline['start_time'].iloc[0], periods=total_seconds, freq='s')
        
        df_micro = pd.DataFrame({
            'timestamp': timestamps,
            'state': state_series[:total_seconds],
            'W_effective': weather_global[:total_seconds],
            'P_main_kW': p_main_physical + fuzz_main,
            'P_aux_kW': p_aux_physical + fuzz_aux,
            'W_cut_in': w_cut_in,
            'W_saturation': w_saturation,
            'W_baseline': w_mean
        })
        return df_micro