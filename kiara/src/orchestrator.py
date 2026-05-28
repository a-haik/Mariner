import pandas as pd
import numpy as np
from core.dispatcher import FerryDispatcher
from core.power import PowerGenerator
from config.vessel_specs import ATMOSPHERE_CONFIG

class KiaraOrchestrator:
    def __init__(self, schedule, start_dt, delay_params, power_config, weather_params, k_factors, initial_weather=None):
        self.dispatcher = FerryDispatcher(schedule, start_dt, delay_params)
        self.power_gen = PowerGenerator(power_config)
        self.weather_params = weather_params
        self.k_factors = k_factors
        self.initial_weather = initial_weather
        
    def _generate_turbulent_weather(self, num_seconds, gust_amp_fraction):
        """
        Generates a composite environmental weather field combining long-term
        Jacobi climatology with multi-scale Ornstein-Uhlenbeck wind gust layers.
        """
        dt = 1.0 / 3600.0  # Hourly increments for Jacobi process
        dt_sec = 1.0       # 1Hz increments for high-frequency gusts
        
        # 1. Base Climatological Macro Layer (Jacobi)
        mu = self.weather_params['mu']
        theta = self.weather_params['theta']
        sigma = self.weather_params['sigma']
        
        W_clima = np.zeros(num_seconds)
        W_clima[0] = np.clip(self.initial_weather, 0.001, 0.999) if self.initial_weather is not None else np.clip(np.random.normal(mu, 0.1), 0.05, 0.95)
        
        for t in range(1, num_seconds):
            drift = theta * (mu - W_clima[t-1]) * dt
            diffusion = sigma * np.sqrt(W_clima[t-1] * (1.0 - W_clima[t-1])) * np.random.normal(0, np.sqrt(dt))
            W_clima[t] = np.clip(W_clima[t-1] + drift + diffusion, 0.001, 0.999)
            
        # 2. Add Turbulent Meso/Micro Wind Layers via OU processes
        g_slow = np.zeros(num_seconds)
        g_fast = np.zeros(num_seconds)
        
        theta_slow = 1.0 / ATMOSPHERE_CONFIG['tau_gust_slow']
        theta_fast = 1.0 / ATMOSPHERE_CONFIG['tau_gust_fast']
        
        theta_slow = 1.0 / ATMOSPHERE_CONFIG['tau_gust_slow']
        theta_fast = 1.0 / ATMOSPHERE_CONFIG['tau_gust_fast']
        
        # Target steady-state standard deviations from your dashboard slider
        sigma_norm_slow = ATMOSPHERE_CONFIG['sigma_gust_slow_base'] * gust_amp_fraction
        sigma_norm_fast = ATMOSPHERE_CONFIG['sigma_gust_fast_base'] * gust_amp_fraction
        
        # Mathematically scale the per-second diffusion coefficients (OU exact calibration)
        sig_slow = sigma_norm_slow * np.sqrt(2.0 * theta_slow)
        sig_fast = sigma_norm_fast * np.sqrt(2.0 * theta_fast)
        
        dW_slow = np.random.normal(0, np.sqrt(dt_sec), num_seconds)
        dW_fast = np.random.normal(0, np.sqrt(dt_sec), num_seconds)
        
        for t in range(1, num_seconds):
            g_slow[t] = g_slow[t-1] - theta_slow * g_slow[t-1] * dt_sec + sig_slow * dW_slow[t]
            g_fast[t] = g_fast[t-1] - theta_fast * g_fast[t-1] * dt_sec + sig_fast * dW_fast[t]
            
        # Combine macro baseline with structural atmospheric deviations
        W_effective = np.clip(W_clima + g_slow + g_fast, 0.001, 0.999)
        return W_effective

    def run_simulation(self, tunable_params, days=1, enable_delays=True, departure_hour=9):
        padded_seconds = (days + 1) * 24 * 3600
        
        # Extract dashboard 'gust_amp_fraction' slider safely 
        gust_amp_fraction = tunable_params.get('gust_amp_fraction', 0.5)
        weather_global = self._generate_turbulent_weather(padded_seconds, gust_amp_fraction)
        
        df_timeline = self.dispatcher.generate_timeline(
            weather_array=weather_global, 
            days=days, 
            enable_delays=enable_delays,
            morning_departure_hour=departure_hour
        )
        
        month_mu = self.weather_params['mu']
        df_micro = self.power_gen.generate_traces(
            df_timeline=df_timeline, 
            weather_global=weather_global, 
            k_factors=self.k_factors, 
            tunable_params=tunable_params,
            month_mu=month_mu,
            dt_seconds=1
        )
        
        return df_timeline, df_micro