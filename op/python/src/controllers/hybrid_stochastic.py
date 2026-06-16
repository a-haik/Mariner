# python/src/controllers/hybrid_stochastic.py
import numpy as np
from numba import njit
from config import SimConfig

@njit(cache=True)
def _lookup_hybrid_policy(t_k: int, pd_val: float, n_prev: int, soc_val: float,
                          p_vals: np.ndarray, n_vals: np.ndarray, soc_vals_grid: np.ndarray,
                          policy_n: np.ndarray, policy_pfc: np.ndarray) -> tuple[int, float]:
    """
    JIT-compiled closed-loop policy extractor.
    Maps current physical states to the optimal pre-calculated Bellman actions.
    """
    # 1. Exact/Nearest Match for Discrete States (P_d and n)
    idx_p = np.abs(p_vals - pd_val).argmin()
    idx_n = np.abs(n_vals - n_prev).argmin()
    
    # 2. State of Charge Policy Mapping
    # Clamp SoC to grid boundaries to prevent indexing errors
    soc_clamped = max(soc_vals_grid[0], min(soc_vals_grid[-1], soc_val))
    
    # For discrete action 'n', use nearest neighbor to avoid fractional modules
    idx_soc_nearest = np.abs(soc_vals_grid - soc_clamped).argmin()
    best_n = policy_n[t_k, idx_p, idx_n, idx_soc_nearest]
    
    # For continuous action 'P_fc', use 1D linear interpolation
    idx_soc = np.searchsorted(soc_vals_grid, soc_clamped) - 1
    # Edge case handling if it hits exactly the top boundary
    if idx_soc == len(soc_vals_grid) - 1:
        best_pfc = policy_pfc[t_k, idx_p, idx_n, idx_soc]
    else:
        s0, s1 = soc_vals_grid[idx_soc], soc_vals_grid[idx_soc + 1]
        v0 = policy_pfc[t_k, idx_p, idx_n, idx_soc]
        v1 = policy_pfc[t_k, idx_p, idx_n, idx_soc + 1]
        weight = (soc_clamped - s0) / (s1 - s0)
        best_pfc = v0 + weight * (v1 - v0)
        
    return best_n, best_pfc

class HybridStochasticControl:
    """
    Online execution agent for the Multi-Timescale system.
    Queried in real-time by the Simulator at every macro-step.
    """
    def __init__(self, config: SimConfig, p_vals: np.ndarray, 
                 policy_n: np.ndarray, policy_pfc: np.ndarray):
        self.config = config
        self.p_vals = p_vals
        self.policy_n = policy_n
        self.policy_pfc = policy_pfc
        
        # Reconstruct the continuous grid used during solver training
        self.soc_grid = np.arange(0.0, 100.0 + self.config.soc_step, self.config.soc_step)

    def get_action(self, macro_step_k: int, current_pd: float, 
                   n_prev: int, current_soc: float) -> tuple[int, float]:
        """Returns the optimal (n_k, P_fc_k) given the current physical states."""
        return _lookup_hybrid_policy(
            macro_step_k, current_pd, n_prev, current_soc,
            self.p_vals, self.config.n_vals, self.soc_grid,
            self.policy_n, self.policy_pfc
        )