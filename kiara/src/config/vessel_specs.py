import numpy as np

CLIMATE_STATS = {
    'W_annual_mean': 0.09142,      
    'W_annual_std': 0.06056,       
}

POWER_CONFIG = {
    # --- Physical Inertia Response Metrics ---
    'tau_diesel': 15.0,            # Main engine response lag (seconds)
    'tau_electric': 1.0,           # PM Motor response lag (seconds)
    'tau_human': 8.0,              # Bridge handle hold governor lag (seconds)

    # --- Mode-Specific Steady-State Benchmarks (kW) ---
    'P_main_transit': 21240,       # Open sea cruise propulsion
    'P_aux_transit': 210,          # Open sea base hotel load
    
    'P_main_maneuver_base': 5000,  # Harbor maneuvering baseline
    'P_main_maneuver_max': 5400,   # Harbor maneuvering max wind fighting peak
    'P_aux_maneuver_base': 120,    # Harbor maneuvering base electric load
    'P_aux_thruster_max': 350,     # Bow thruster electric max draw

    'P_main_port_ops': 5000,        # Main engine load during vehicle loading
    'P_aux_port_ops': 120,         # Ramp hydraulics / hotel load in port

    'P_main_idling': 0,            # Main engines completely shut down
    'P_aux_idling': 120,           # Baseline auxiliary/hotel load (moored)

    # --- Structural Overrides & Safety Bounds ---
    'aux_volatility_reduction': 0.25,
    'alpha_thruster_start': 1.0,   
    'beta_thruster_max': 2.0       
}


ATMOSPHERE_CONFIG = {
    'tau_gust_slow': 600.0,        # Meso-scale wind wave trend memory (10 mins)
    'tau_gust_fast': 30.0,         # Micro-scale aerodynamic turbulence memory (30 secs)
    'sigma_gust_slow_base': 0.03,  # Fixed physical amplitude of slow drift variations
    'sigma_gust_fast_base': 0.05,  # Fixed physical amplitude of sharp gust variations
}

STATE_PHYSICS = {
    'transit': {
        'tau': 180.0,              # Long wave swells
        'sigma_multiplier': 1.0
    },
    'maneuvering': {
        'tau': 10.0,               # Human throttle adjustments
        'sigma_multiplier': 2.0
    },
    'port_operations': {
        'tau': 60.0,
        'sigma_multiplier': 0.25    # Pure instrument static applies
    },
    'idling': {
        'tau': 300.0,
        'sigma_multiplier': 0.1
    }
}

DELAY_PARAMS = {
    'transit': {'mu': 0.5, 'sigma': 0.5},
    'port': {'mu': 1.0, 'sigma': 0.8},
    'maneuvering': {'avg_mins': 6.0, 'mu': 0.0, 'sigma': 0.5},
    'weather_penalty': 0.5
}