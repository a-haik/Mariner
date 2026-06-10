# python/src/plants/hybrid_plant.py
import numpy as np
from numba import njit
from config import SimConfig
from src.plants.physics import calculate_fc_cost_per_second

@njit(cache=True)
def _get_pfc_bounds(soc_current: float, pd_current: float, e_bat_kwh: float, dt: float) -> tuple[float, float]:
    """
    Tier 1 Action-Space Pruning: Calculates the exact physical boundaries 
    for Fuel Cell power (P_fc) to guarantee the battery stays within 0-100% SoC
    over the next micro-step.
    """
    # kw_capacity is the maximum continuous power the battery can absorb/discharge 
    # over one dt step to swing exactly 100% of its capacity.
    kw_capacity = (e_bat_kwh * 3600.0) / dt
    
    p_fc_min = pd_current - (soc_current / 100.0) * kw_capacity
    p_fc_max = pd_current + ((100.0 - soc_current) / 100.0) * kw_capacity
    
    return p_fc_min, p_fc_max

@njit(cache=True)
def _simulate_micro_physics(soc_k: float, n_k: int, n_prev: int, p_fc_k: float, 
                            p_d_micro: np.ndarray, dt: float, e_bat: float, 
                            c_bat_kwh: float, n_eol: int, p_star: float, 
                            k_s: float, penalty_wall: float,
                            # --- New Config Arguments ---
                            k_h2: float, k_fc: float, tau_fc: float, 
                            a0: float, a1: float, a2: float, alpha_deg: float) -> tuple[float, float, float, float]:
    
    lambda_scale = len(p_d_micro)
    soc_t = soc_k
    accumulated_c_bat = 0.0
    
    # --- 1. True Macro-scale FC Costs ---
    if n_k > 0:
        p_module = p_fc_k / n_k
        c_o_sec = calculate_fc_cost_per_second(p_module, p_star, k_h2, k_fc, tau_fc, a0, a1, a2, alpha_deg)
        c_o = n_k * c_o_sec * (dt * lambda_scale)
    else:
        c_o = penalty_wall
        
    c_s = k_s * abs(n_k - n_prev)
    
    # --- 2. Micro-scale Battery Simulation Loop ---
    for t in range(lambda_scale):
        p_bat_t = p_d_micro[t] - p_fc_k
        delta_soc = - (p_bat_t * (dt / 3600.0) / e_bat) * 100.0
        soc_t += delta_soc
        
        # Keep penalty wall active for the optimizer's sake
        if soc_t < 0.0 or soc_t > 100.0:
            return penalty_wall, penalty_wall, penalty_wall, soc_k
            
        c_bat_step = (abs(p_bat_t) * (dt / 3600.0) * e_bat * c_bat_kwh) / (2.0 * n_eol * e_bat)
        accumulated_c_bat += c_bat_step
        
    total_cost = c_o + c_s + accumulated_c_bat
    
    return total_cost, c_o, c_s, soc_t

class FuelCellBatteryPlant:
    """
    Multi-timescale physical simulator for the Hybrid PEMFC/Battery vessel.
    """
    def __init__(self, config: SimConfig):
        self.config = config

    def get_pruned_action_space(self, soc_current: float, pd_current: float) -> tuple[float, float]:
        """Wrapper to retrieve exact bounds for solver Tier 1 Pruning."""
        return _get_pfc_bounds(soc_current, pd_current, self.config.e_bat, self.config.dt)

    def calculate_macro_step(self, soc_k: float, n_k: int, n_prev: int, p_fc_k: float, 
                        p_d_micro: np.ndarray) -> tuple[float, float]:
        """Runs the Numba physics loop and returns (total_cost, final_soc)."""
        return _simulate_micro_physics(
            soc_k=soc_k, n_k=n_k, n_prev=n_prev, p_fc_k=p_fc_k,
            p_d_micro=p_d_micro, dt=self.config.dt, e_bat=self.config.e_bat,
            c_bat_kwh=self.config.c_bat_kwh, n_eol=self.config.n_eol_cycles,
            p_star=self.config.p_star, k_s=self.config.k_s, 
            penalty_wall=self.config.penalty_wall,
            k_h2=self.config.k_h2, k_fc=self.config.k_fc, 
            tau_fc=self.config.tau_fc, a0=self.config.a0, 
            a1=self.config.a1, a2=self.config.a2, 
            alpha_deg=self.config.alpha_deg
        )