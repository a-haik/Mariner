import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# Assuming these are saved in your kiara/src/core/ directory
from core.dispatcher import FerryDispatcher
from core.power import PowerGenerator

class KiaraOrchestrator:
    def __init__(self, schedule, start_dt, delay_params, power_config, weather_params, k_factors, initial_weather=None):
        """Initializes the full KIARA synthetic data pipeline."""
        self.dispatcher = FerryDispatcher(schedule, start_dt, delay_params)
        self.power_gen = PowerGenerator(power_config)
        self.weather_params = weather_params
        self.k_factors = k_factors
        self.initial_weather = initial_weather
        
    def _generate_jacobi_weather(self, num_seconds):
        """
        Generates the Global Meso Weather using the bounded Jacobi SDE.
        (Euler-Maruyama numerical integration)
        """
        mu = self.weather_params['mu']
        theta = self.weather_params['theta']
        sigma = self.weather_params['sigma']
        dt = 1.0 / 3600.0  # Time step in hours (since theta is per hour)
        
        W = np.zeros(num_seconds)
        
        # Random initial state around the mean OR custom UI input
        if self.initial_weather is None: # Modifié ici (PEP 8)
            W[0] = np.clip(np.random.normal(mu, 0.1), 0.05, 0.95)
        else:
            # Sécurité ajoutée ici pour éviter les erreurs mathématiques aux extrêmes
            W[0] = np.clip(self.initial_weather, 0.001, 0.999) 
        
        for t in range(1, num_seconds):
            # Jacobi Drift
            drift = theta * (mu - W[t-1]) * dt
            # Jacobi Diffusion (vanishes at 0 and 1)
            diffusion = sigma * np.sqrt(W[t-1] * (1 - W[t-1])) * np.random.normal(0, np.sqrt(dt))
            
            # Update and strictly bound to prevent float precision errors
            W[t] = np.clip(W[t-1] + drift + diffusion, 0.001, 0.999)
            
        return W

    def run_simulation(self, days=1):
        print(f"--- Starting KIARA Simulation ({days} Days) ---")
        
        # 1. Pre-generate Meso Weather for a safely padded timeframe (e.g., days + 1)
        print("1. Generating Meso Weather (Jacobi SDE)...")
        padded_seconds = (days + 1) * 24 * 3600
        weather_global = self._generate_jacobi_weather(padded_seconds)
        
        # 2. Macro Layer (Topology) - Now we pass the weather to the dispatcher
        print("2. Generating Macro Dispatcher Timeline...")
        df_timeline = self.dispatcher.generate_timeline(days=days, weather_array=weather_global)
        
        # Trim weather array to exactly match the generated timeline length
        total_seconds = int((df_timeline['end_time'].max() - df_timeline['start_time'].min()).total_seconds())
        weather_global = weather_global[:total_seconds]
        
        # 3. Micro Layer (Power)
        print("3. Generating Micro Bivariate Power Traces...")
        df_micro = self.power_gen.generate_traces(df_timeline, weather_global, self.k_factors, dt_seconds=1)
        
        df_micro['W_global'] = weather_global
        print("Simulation Complete!")
        return df_timeline, df_micro