POWER_CONFIG = {
    # Physical Inertia (seconds)
    'tau_diesel': 15.0,     
    'tau_electric': 1.0,    
    
    # Baseline Loads (kW)
    'P_main_sea': 21240,    
    'P_main_port': 120,
    'P_main_maneuver':5000,     
    'P_aux_sea': 210,       
    'P_aux_port': 120,
    'P_aux_maneuver': 210,      
    'P_aux_hotel': 50,      
    
    # Weather Dynamics & Noise
    'sigma_main_base': 500, 
    'sigma_aux_base': 15,              # NEW: Normal fluctuation of hotel loads (kW)
    
    # Proportional Thruster Logic
    'thruster_weather_threshold': 0.15, # Wind intensity where thrusters become necessary
    'thruster_max_weather': 0.30,       # Wind intensity requiring 100% thruster power
    'P_aux_spike_max': 350              # Absolute max bow thruster draw (kW)
}

DELAY_PARAMS = {
    'transit': {'mu': 0.5, 'sigma': 0.5},
    'port':    {'mu': 1.0, 'sigma': 0.8},
    'maneuvering': {'avg_mins': 5.0, 'mu': 0, 'sigma': 0.8}
}