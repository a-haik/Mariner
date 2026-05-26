import numpy as np
import pandas as pd
from scipy.signal import lfilter
from config.vessel_specs import CLIMATE_STATS

class PowerGenerator:
    def __init__(self, config):
        self.config = config
        
    def _apply_inertia(self, target_series, tau, dt=1.0):
        alpha = np.exp(-dt / tau)
        return lfilter([1.0 - alpha], [1.0, -alpha], target_series)

    def _generate_proportional_ou_noise(self, num_seconds, p_commanded, tau_reversion, sigma_fraction):
        dt = 1.0
        theta = 1.0 / tau_reversion
        sigma_t = p_commanded * sigma_fraction
        
        chi = np.zeros(num_seconds)
        for t in range(1, num_seconds):
            drift = -theta * chi[t-1] * dt
            diffusion = sigma_t[t-1] * np.sqrt(2.0 * theta) * np.random.normal(0, np.sqrt(dt))
            chi[t] = chi[t-1] + drift + diffusion
            
        return chi

    def generate_traces(self, df_timeline, weather_global, k_factors, tunable_params, month_mu, dt_seconds=1):
        """
        tunable_params contains: 'wave_resistance_factor', 'sigma_fraction', 
                                 'gust_amp_fraction', 'delta_instrument'
        """
        total_seconds = int((df_timeline['end_time'].max() - df_timeline['start_time'].min()).total_seconds())
        time_index = pd.date_range(start=df_timeline['start_time'].min(), periods=total_seconds, freq=f'{dt_seconds}s')
        
        df_micro = pd.DataFrame({'timestamp': time_index})
        df_micro = pd.merge_asof(df_micro, df_timeline[['start_time', 'state', 'location']], 
                                 left_on='timestamp', right_on='start_time', direction='backward')
        
        # 1. Multi-Layer Environmental Core Compilation
        k_array = df_micro['location'].map(k_factors).fillna(1.0).values
        W_base = weather_global[:total_seconds] * k_array
        
        slow_gusts = self._apply_inertia(
            np.random.normal(0, self.config['sigma_slow_gust_base'], total_seconds), 
            tau=self.config['tau_gust_slow']
        )
        fast_gusts = self._apply_inertia(
            np.random.normal(0, month_mu * tunable_params['gust_amp_fraction'], total_seconds), 
            tau=self.config['tau_gust_fast']
        )
        W_eff = np.clip(W_base + slow_gusts + fast_gusts, 0.0, 1.0)
        states = df_micro['state'].values
        
        # 2. Derive Rigorous Statistical Corridor Coordinates
        W_baseline = CLIMATE_STATS['W_annual_mean']
        W_cut_in = CLIMATE_STATS['W_annual_mean'] + (self.config['alpha_thruster_start'] * CLIMATE_STATS['W_annual_std'])
        W_sat = CLIMATE_STATS['W_annual_mean'] + (self.config['beta_thruster_max'] * CLIMATE_STATS['W_annual_std'])
        
        target_main = np.zeros(total_seconds)
        target_hotel = np.zeros(total_seconds)     
        target_thruster = np.zeros(total_seconds)  
        
        for t in range(total_seconds):
            w = W_eff[t]
            st = states[t]
            
            if st == 'transit':
                delta_w = max(0.0, w - W_baseline)
                target_main[t] = self.config['P_main_sea'] * (1.0 + (tunable_params['wave_resistance_factor'] * delta_w))
                target_hotel[t] = self.config['P_aux_sea']
                target_thruster[t] = 0.0 
                
            elif st == 'maneuvering':
                target_hotel[t] = self.config['P_aux_maneuver']
                if w <= W_cut_in:
                    target_main[t] = self.config['P_main_maneuver']
                    target_thruster[t] = 0.0
                else:
                    gamma = np.clip((w - W_cut_in) / (W_sat - W_cut_in), 0.0, 1.0)
                    target_main[t] = self.config['P_main_maneuver'] + gamma * (self.config['P_main_port_max'] - self.config['P_main_maneuver'])
                    target_thruster[t] = gamma * (self.config['P_aux_spike_max'] - self.config['P_aux_maneuver'])
                    
            elif st == 'port_dwell':
                target_hotel[t] = self.config['P_aux_port_base']
                if w <= W_cut_in:
                    target_main[t] = self.config['P_main_port_base']
                    target_thruster[t] = 0.0
                else:
                    gamma = np.clip((w - W_cut_in) / (W_sat - W_cut_in), 0.0, 1.0)
                    target_main[t] = self.config['P_main_port_base'] + gamma * (self.config['P_main_port_max'] - self.config['P_main_port_base'])
                    target_thruster[t] = gamma * (self.config['P_aux_spike_max'] - self.config['P_aux_port_base'])
                    
            elif st == 'overnight_dwell':
                target_main[t] = 0.0
                target_hotel[t] = self.config['P_aux_hotel']
                target_thruster[t] = 0.0

        # 3. Apply Systems Inertia
        p_main_commanded = self._apply_inertia(target_main, tau=self.config['tau_diesel'])
        p_hotel_commanded = self._apply_inertia(target_hotel, tau=self.config['tau_electric'])
        p_thruster_commanded = self._apply_inertia(target_thruster, tau=self.config['tau_human'])
        
        # 4. Generate Decoupled Proportional OU Noise Multipliers
        ou_noise_main = self._generate_proportional_ou_noise(total_seconds, p_main_commanded, self.config['tau_diesel'], tunable_params['sigma_fraction'])
        ou_noise_hotel = self._generate_proportional_ou_noise(total_seconds, p_hotel_commanded, self.config['tau_electric'], tunable_params['sigma_fraction'] * self.config['aux_volatility_reduction'])
        ou_noise_thruster = self._generate_proportional_ou_noise(total_seconds, p_thruster_commanded, self.config['tau_electric'], tunable_params['sigma_fraction'])
        
        # 5. Assembly
        p_main_noisy = p_main_commanded + ou_noise_main
        p_thruster_noisy = np.clip(p_thruster_commanded + ou_noise_thruster, 0.0, None)
        p_aux_noisy = (p_hotel_commanded + ou_noise_hotel) + p_thruster_noisy

        p_main_noisy = np.maximum(0.0, p_main_noisy)
        p_aux_noisy = np.maximum(0.0, p_aux_noisy)
        
        # 6. Inject Telemetry Instrument Error via Dashboard Variable Control
        delta_inst = tunable_params['delta_instrument']
        fuzz_main = np.random.normal(0, delta_inst * p_main_noisy)
        fuzz_aux = np.random.normal(0, delta_inst * p_aux_noisy)
        
        df_micro['P_main_kW'] = np.clip(p_main_noisy + fuzz_main, 0.0, None)
        df_micro['P_aux_kW'] = np.clip(p_aux_noisy + fuzz_aux, 0.0, None)
        df_micro['W_effective'] = W_eff
        df_micro['W_cut_in'] = W_cut_in
        df_micro['W_saturation'] = W_sat
        df_micro['W_baseline'] = W_baseline
        
        return df_micro