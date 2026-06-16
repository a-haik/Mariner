# src/plants/physics.py
from numba import njit

@njit(cache=True)
def calculate_fc_cost_per_second(p_module: float, p_nom: float, 
                                 k_h2: float, k_fc: float, tau_fc: float, 
                                 a0: float, a1: float, a2: float, alpha_deg: float) -> float:
    """Evaluates the physical cost of running a single FC module for one second."""
    
    # 1. Hydrogen Consumption (Polynomial yields g/s)
    m_H2_g_s = a0 + a1 * p_module + a2 * (p_module ** 2)
    cost_h2_sec = k_h2 * (m_H2_g_s / 1000.0)
    
    # 2. Degradation (Fraction of life lost per second)
    d_FC_sec = (1.0 / (3600.0 * tau_fc)) * (1.0 + alpha_deg * ((p_module - p_nom)**2) / (p_nom**2))
    cost_deg_sec = k_fc * d_FC_sec
    
    return cost_h2_sec + cost_deg_sec