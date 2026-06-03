# src/solvers/dp_multiscale.py
import numpy as np
from numba import njit

@njit(cache=True)
def _solve_bellman_recursion(T: int, p_vals: np.ndarray, n_vals: np.ndarray, 
                             transition_matrix: np.ndarray, k_s: float, p_star: float) -> np.ndarray:
    """
    JIT-compiled backward induction routine solving the discrete Bellman recursion.
    Replicates original MATLAB logic bug-for-bug to ensure numerical validation.
    """
    p_size = len(p_vals)
    n_size = len(n_vals)
    
    V = np.full((T, p_size, n_size), np.inf, dtype=np.float64)
    policy = np.zeros((T, p_size, n_size), dtype=np.int32)
    
    # 1. Exact MATLAB Terminal Condition (t = T-1 in Python)
    for i in range(p_size):
        p_val = p_vals[i]
        for j in range(n_size):
            n_val = n_vals[j]
            if n_val > 0:
                V[T - 1, i, j] = ((p_val / p_star - n_val) ** 2) / n_val
                
    # 2. Exact MATLAB Backward Iteration (T-2 down to 0)
    for t in range(T - 2, -1, -1):
        for i in range(p_size):
            p_val = p_vals[i]
            for j in range(n_size):
                n_val = n_vals[j]
                if n_val <= 0:
                    continue
              
                # MATLAB Bug 1: Calculating Operational Cost OUTSIDE the action loop, 
                # using the INHERITED state (n_val) instead of the action state (n_next).
                C_o = ((p_val / p_star - n_val) ** 2) / n_val
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
        """Generates and returns the lookup policy array."""
        return _solve_bellman_recursion(
            T=horizon_length,
            p_vals=self.mc_model['levels'],
            n_vals=self.config.n_vals,
            transition_matrix=self.mc_model['P'],
            k_s=self.config.k_s,
            p_star=self.config.p_star
        )