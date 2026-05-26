import numpy as np
import pandas as pd
from scipy.signal import lfilter

class PowerGenerator:
    def __init__(self, config):
        """
        Initializes the power generator with physical and operational parameters.
        :param config: Dictionary containing inertia (tau) and baseline power levels.
        """
        self.config = config
        
    def _apply_inertia(self, target_series, tau, dt=1.0):
        """
        Applies a First-Order Low-Pass Filter (Euler-Maruyama for OU process).
        This smooths instantaneous state changes, simulating physical engine spool-up.
        
        :param target_series: The raw, blocky step-function of target power.
        :param tau: Relaxation time (seconds). High for diesel, low for electric.
        :param dt: Time step (seconds).
        """
        # Calculate the filter coefficient alpha = e^(-dt / tau)
        alpha = np.exp(-dt / tau)
        
        # Scipy's lfilter is infinitely faster than a Python for-loop
        # y[n] = (1 - alpha) * x[n] + alpha * y[n-1]
        b = [1 - alpha]
        a = [1, -alpha]
        
        smoothed_power = lfilter(b, a, target_series)
        return smoothed_power

    def _generate_thruster_spikes(self, weather_series, state_series):
        """
        Generates localized power spikes for the electric bow thrusters.
        Spike amplitude scales proportionally with weather severity.
        """
        spikes = np.zeros(len(state_series))
        is_port = (state_series == 'port_dwell')
        
        # Calculate proportional wind intensity [0.0 to 1.0]
        # 0.0 means at or below threshold. 1.0 means at or above max weather.
        wind_intensity = np.clip(
            (weather_series - self.config['thruster_weather_threshold']) / 
            (self.config['thruster_max_weather'] - self.config['thruster_weather_threshold']), 
            0, 1
        )
        
       # Thrusters are active during BOTH Port Dwell AND Maneuvering
        is_active = (state_series == 'port_dwell') | (state_series == 'maneuvering')
        
        wind_intensity = np.clip(
            (weather_series - self.config['thruster_weather_threshold']) / 
            (self.config['thruster_max_weather'] - self.config['thruster_weather_threshold']), 
            0, 1
        )
        active_indices = is_active & (wind_intensity > 0)
        
        # The maximum possible spike at time t scales with the wind intensity
        max_spike_at_t = wind_intensity * self.config['P_aux_spike_max']
        
        # Draw random spike amplitudes between 0 and the current maximum
        spike_amplitudes = np.random.uniform(0, max_spike_at_t)
        
        # Simulate operator "bursts" (approx 10% duty cycle when active)
        duty_cycle_mask = np.random.rand(len(spikes)) < 0.10 
        
        # Apply the spikes
        mask = active_indices & duty_cycle_mask
        spikes[mask] = spike_amplitudes[mask]
        
        return spikes

    def generate_traces(self, df_timeline, weather_global, k_factors, dt_seconds=1):
        """
        Generates the 1-second resolution power profiles with spatial weather adjustments.
        """
        print("Expanding macro timeline to high-frequency micro traces...")
        
        total_seconds = int((df_timeline['end_time'].max() - df_timeline['start_time'].min()).total_seconds())
        time_index = pd.date_range(start=df_timeline['start_time'].min(), periods=total_seconds, freq=f'{dt_seconds}s')
        
        df_micro = pd.DataFrame({'timestamp': time_index})
        
        # 1. Map states AND locations to the 1-second dataframe
        df_micro = pd.merge_asof(df_micro, df_timeline[['start_time', 'state', 'location']], 
                                 left_on='timestamp', right_on='start_time', direction='backward')
        
        # 2. Map baseline power levels
        state_to_main = {'transit': self.config['P_main_sea'], 
                         'maneuvering': self.config['P_main_maneuver'], # <--- NEW
                         'port_dwell': self.config['P_main_port'],
                         'overnight_dwell': 0}
                         
        state_to_aux = {'transit': self.config['P_aux_sea'],
                        'maneuvering': self.config['P_aux_maneuver'], 
                        'port_dwell': self.config['P_aux_port'],
                        'overnight_dwell': self.config['P_aux_hotel']}
        
        target_main = df_micro['state'].map(state_to_main).fillna(0).values
        target_aux = df_micro['state'].map(state_to_aux).fillna(0).values
        
        # 3. Weather Integration (Spatial Multiplier + Gusts)
        k_array = df_micro['location'].map(k_factors).fillna(1.0).values
        W_base = weather_global * k_array
        
        # Overlay short-term Gusts (OU process, 10 min relaxation)
        raw_gusts = np.random.normal(0, 0.15, size=len(W_base))
        smoothed_gusts = self._apply_inertia(raw_gusts, tau=600)
        W_eff = np.clip(W_base + smoothed_gusts, 0, 1)
        
        # --- Wave Resistance Penalty (Baseline increases in bad weather) ---
        wave_penalty = self.config.get('wave_resistance_factor', 0.0)
        target_main_adjusted = target_main * (1.0 + (wave_penalty * W_eff))
        
        # Add stochastic noise to MAIN engines
        noise_main = np.random.normal(0, self.config['sigma_main_base'] * (1 + W_eff))
        target_main_noisy = np.clip(target_main_adjusted + noise_main, 0, None)
        
        # Add baseline stochastic noise to AUX engines (Hotel loads)
        noise_aux = np.random.normal(0, self.config['sigma_aux_base'], size=len(target_aux))
        target_aux_noisy = np.clip(target_aux + noise_aux, 0, None)
        
        # 4. Apply Physical Engine Inertia
        df_micro['P_main_kW'] = self._apply_inertia(target_main_noisy, tau=self.config['tau_diesel'])
        
        # 5. Build Aux Power (Noisy Baseline + Proportional Thrusters)
        thruster_spikes = self._generate_thruster_spikes(W_eff, df_micro['state'].values)
        target_aux_total = target_aux_noisy + thruster_spikes
        df_micro['P_aux_kW'] = self._apply_inertia(target_aux_total, tau=self.config['tau_electric'])
        
        return df_micro

# ==========================================
# Example Configuration for the Expert Meeting
# ==========================================
if __name__ == "__main__":
    # This config is what you tweak live during the meeting
    config = {
        # Physical Inertia (seconds)
        'tau_diesel': 15.0,     # Takes 15s to spool up/down
        'tau_electric': 1.0,    # Near instant response
        
        # Baselines (kW)
        'P_main_sea': 21240,
        'P_main_port': 120,
        'P_aux_sea': 210,
        'P_aux_port': 120,
        'P_aux_hotel': 50,      # Overnight load
        
        # Weather Dynamics
        'sigma_main_base': 500, # Base noise amplitude 
        'thruster_weather_threshold': 0.6, # W(t) > 0.6 triggers thrusters
        'P_aux_spike_max': 350  # Max bow thruster draw
    }