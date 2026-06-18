# src/solvers/sdp_baseline.py
import numpy as np
from numba import njit
from src.plants.physics import calculate_fc_cost_per_second

@njit(cache=True)
def _solve_bellman_recursion(T: int, p_vals: np.ndarray, n_vals: np.ndarray, 
                             transition_matrix: np.ndarray, k_s: float, p_nom: float, p_max: float,
                             dt_macro: float, k_h2: float, k_fc: float, tau_fc: float, 
                             a0: float, a1: float, a2: float, alpha_deg: float) -> np.ndarray:
    
    p_size = len(p_vals)
    n_size = len(n_vals)
    
    V = np.full((T, p_size, n_size), np.inf, dtype=np.float64)
    policy = np.zeros((T, p_size, n_size), dtype=np.int32)
    
    # 1. Terminal Condition
    for i in range(p_size):
        p_val = p_vals[i]
        for j in range(n_size):
            n_val = n_vals[j]
            if n_val > 0:
                if (p_val / n_val) > p_max:
                    V[T - 1, i, j] = np.inf
                else:
                    c_o_sec = calculate_fc_cost_per_second(p_val / n_val, p_nom, k_h2, k_fc, tau_fc, a0, a1, a2, alpha_deg)
                    V[T - 1, i, j] = n_val * c_o_sec * dt_macro
                
    # 2. Backward Iteration
    for t in range(T - 2, -1, -1):
        for i in range(p_size):
            p_val = p_vals[i]
            for j in range(n_size):
                n_val = n_vals[j] # Following draft parity: evaluating on n(t-1)
                
                if n_val > 0:
                    if (p_val / n_val) > p_max:
                        C_o = np.inf
                    else:
                        c_o_sec = calculate_fc_cost_per_second(p_val / n_val, p_nom, k_h2, k_fc, tau_fc, a0, a1, a2, alpha_deg)
                        C_o = n_val * c_o_sec * dt_macro
                else:
                    C_o = np.inf
                
                best_cost = np.inf
                best_action_idx = 0  
                
                for a_idx in range(n_size):
                    n_next = n_vals[a_idx]
                    C_s = k_s * abs(n_next - n_val)
                    
                    exp_future = 0.0
                    for i_next in range(p_size):
                        exp_future += transition_matrix[i, i_next] * V[t + 1, i_next, a_idx]
                        
                    total_cost = C_o + C_s + exp_future
                    if total_cost < best_cost:
                        best_cost = total_cost
                        best_action_idx = a_idx  
                        
                V[t, i, j] = best_cost
                policy[t, i, j] = best_action_idx
                
    return policy

class BaselineSDPSolver:
    """
    Orchestrates the offline generation of the Bellman policy matrix
    without running any time-series simulation.
    """
    def __init__(self, config, mc_model):
        self.config = config
        self.mc_model = mc_model

    def compute_policy_matrix(self, horizon_length: int) -> np.ndarray:
        c = self.config # Alias for readability
        return _solve_bellman_recursion(
            T=horizon_length,
            p_vals=self.mc_model['levels'],
            n_vals=c.n_vals,
            transition_matrix=self.mc_model['P'],
            k_s=c.k_s,
            p_nom=c.p_nom, p_max=c.p_max,
            dt_macro=float(c.Ts), 
            k_h2=c.k_h2, k_fc=c.k_fc, tau_fc=c.tau_fc, 
            a0=c.a0, a1=c.a1, a2=c.a2, alpha_deg=c.alpha_deg
        )