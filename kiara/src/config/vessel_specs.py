import numpy as np

# Ground-truth climate statistics derived from the 30-year Copernicus Aegean Dataset
CLIMATE_STATS = {
    'W_annual_mean': 0.09142,      # Historical yearly average (Clean-hull reference anchor)
    'W_annual_std': 0.06056,       # Historical yearly standard deviation
}

POWER_CONFIG = {
    # Physical Inertia & Memory Timesteps (Seconds) - Rigid Constants
    'tau_diesel': 15.0,        
    'tau_electric': 1.0,       
    'tau_gust_slow': 600.0,    
    'tau_gust_fast': 30.0,     
    'tau_human': 8.0,              # Human-in-the-loop bridge handle hold lag constant
    'tau_drift': 180.0,

    # Structural Atmospheric Noise Constants
    'sigma_slow_gust_base': 0.05,  # Amplitude of meso-scale drift variations
    
    # Statistical Harbor Override Constants (Alpha & Beta Physics Layout)
    'alpha_thruster_start': 1.0,   # Thrusters engage at μ + 1.0σ (Routine windy days)
    'beta_thruster_max': 2.0,      # Full system power maxes out at μ + 2.0σ (Strong gales)

    # Dynamic Power Benchmarks (kW)
    'P_main_sea': 21240,       
    'P_main_port_base': 5000,   
    'P_main_port_max': 5400,   
    'P_main_maneuver': 5000,   
    
    'P_aux_sea': 210,
    'P_aux_port_base': 120,
    'P_aux_maneuver': 120,
    'P_aux_spike_max': 350,    
    'P_aux_hotel': 120,        

    'aux_volatility_reduction': 0.25
}

DELAY_PARAMS = {
    'transit': {'mu': 0.5, 'sigma': 0.5},
    'port': {'mu': 1.0, 'sigma': 0.8},
    'maneuvering': {'avg_mins': 6.0, 'mu': 0.0, 'sigma': 0.5},
    'weather_penalty': 0.5
}