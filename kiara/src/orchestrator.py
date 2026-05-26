import pandas as pd
import numpy as np

from core.dispatcher import FerryDispatcher
from core.power import PowerGenerator

class KiaraOrchestrator:
    def __init__(self, schedule, start_dt, delay_params, power_config, weather_params, k_factors, initial_weather=None):
        self.dispatcher = FerryDispatcher(schedule, start_dt, delay_params)
        self.power_gen = PowerGenerator(power_config)
        self.weather_params = weather_params
        self.k_factors = k_factors
        self.initial_weather = initial_weather
        
    def _generate_jacobi_weather(self, num_seconds):
        mu = self.weather_params['mu']
        theta = self.weather_params['theta']
        sigma = self.weather_params['sigma']
        dt = 1.0 / 3600.0  
        
        W = np.zeros(num_seconds)
        if self.initial_weather is None:
            W[0] = np.clip(np.random.normal(mu, 0.1), 0.05, 0.95)
        else:
            W[0] = np.clip(self.initial_weather, 0.001, 0.999)
        
        for t in range(1, num_seconds):
            drift = theta * (mu - W[t-1]) * dt
            diffusion = sigma * np.sqrt(W[t-1] * (1.0 - W[t-1])) * np.random.normal(0, np.sqrt(dt))
            W[t] = np.clip(W[t-1] + drift + diffusion, 0.001, 0.999)
            
        return W

    def run_simulation(self, tunable_params, days=1, enable_delays=True, departure_hour=9):
        padded_seconds = (days + 1) * 24 * 3600
        weather_global = self._generate_jacobi_weather(padded_seconds)
        
        # Macro Layer Operation absorbs the customized hour anchor
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