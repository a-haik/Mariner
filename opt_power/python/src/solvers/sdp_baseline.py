# python/src/solvers/sdp_baseline.py
import numpy as np
from numba import njit
from typing import Tuple
from src.config import SimConfig

@njit(cache=True)
def _solve_bellman_recursion(T: int, p_vals: np.ndarray, n_vals: np.ndarray, 
                             transition_matrix: np.ndarray, Ts: float, 
                             p_max: float, p_nom: float, k_fc: float, k_h2: float, 
                             S_max: float, tau_fc: float, alpha_fc: float,
                             a0: float, a1: float, a2: float,
                             nT: int, apply_terminal_n_cost: bool):
    """
    JIT-compiled backward induction routine solving the discrete Bellman recursion.
    Updated with true T+1 terminal boundary conditions.
    """
    p_size = len(p_vals)
    n_size = len(n_vals)
    
    # EXPAND V matrix to size T+1. Policy stays T.
    V = np.full((T + 1, p_size, n_size), np.inf, dtype=np.float64)
    policy = np.zeros((T, p_size, n_size), dtype=np.int32)
    k_s = k_fc / S_max

    # Cache for the Expected Future Cost optimization
    exp_future_cache = np.empty(n_size, dtype=np.float64)
    
    # 1. NEW Terminal Boundary Condition (t = T)
    for i in range(p_size):
        for j in range(n_size):
            n_val = n_vals[j]
            if apply_terminal_n_cost:
                V[T, i, j] = k_s * abs(n_val - nT)
            else:
                V[T, i, j] = 0.0

    # 2. Unified Backward Iteration (T-1 down to 0)
    for t in range(T - 1, -1, -1):
        for i in range(p_size):
            p_val = p_vals[i]

            # OPTIMIZATION: Calculate Expected Future Cost for all possible NEXT actions
            for a_idx in range(n_size):
                if n_vals[a_idx] <= 0:
                    continue
                s = 0.0
                for i_next in range(p_size):
                    s += transition_matrix[i, i_next] * V[t + 1, i_next, a_idx]
                exp_future_cache[a_idx] = s
            
            # Loop over current states
            for j in range(n_size):
                n_val = n_vals[j]
                if n_val <= 0:
                    continue
                
                best_cost = np.inf
                best_action_idx = 0  
                
                # Loop over possible ACTIONS (n_next)
                for a_idx in range(n_size):
                    n_next = n_vals[a_idx]
                    if n_next <= 0:
                        continue
                        
                    p_module = p_val / n_next
                    
                    if p_module > p_max:
                        total_cost = np.inf
                    else:
                        exp_future = exp_future_cache[a_idx]

                        m_dot_h2 = a0 + a1 * p_module + a2 * (p_module ** 2)
                        d_fc = (1.0 / (3600.0 * tau_fc)) * (1.0 + alpha_fc * ((p_module - p_nom) ** 2) / (p_nom ** 2))
                        c_o_rate = (k_h2 * m_dot_h2 / 1000.0) + (k_fc * d_fc)
                        
                        C_o = n_next * c_o_rate * Ts
                        C_s = k_s * abs(n_next - n_val)
                            
                        total_cost = C_o + C_s + exp_future
                        
                    if total_cost < best_cost:
                        best_cost = total_cost
                        best_action_idx = a_idx  
                        
                V[t, i, j] = best_cost
                policy[t, i, j] = best_action_idx
                
    return policy, V

class BaselineSDPSolver:
    """Orchestrates the offline generation of the Bellman matrices."""
    def __init__(self, config: SimConfig, mc_model: dict):
        self.config = config
        self.mc_model = mc_model

    def compute_solution(self, horizon_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            policy_matrix: (T, P_size, n_size) array of optimal module counts.
            V_matrix: (T, P_size, n_size) array of expected cumulative costs.
        """
        return _solve_bellman_recursion(
            T=horizon_length,
            p_vals=self.mc_model['levels'],
            n_vals=self.config.n_vals,
            transition_matrix=self.mc_model['P'],
            Ts=float(self.config.Ts),
            p_max=float(self.config.p_max),
            p_nom=float(self.config.p_nom),
            k_fc=float(self.config.k_fc),
            k_h2=float(self.config.k_h2),
            S_max=float(self.config.S_max),
            tau_fc=float(self.config.tau_fc),
            alpha_fc=float(self.config.alpha_fc),
            a0=float(self.config.a0),
            a1=float(self.config.a1),
            a2=float(self.config.a2),
            nT=int(self.config.nT),                              
            apply_terminal_n_cost=bool(self.config.apply_terminal_n_cost)
        )