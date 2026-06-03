# python/src/controllers/stochastic.py
import numpy as np
from numba import njit
from src.controllers.base import ControlLaw

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
                best_action_idx = 0  # Matches MATLAB's initialization: best_action = 1
                
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


@njit(cache=True)
def _forward_pass_lookup(P_d: np.ndarray, p_vals: np.ndarray, n_vals: np.ndarray, 
                         policy: np.ndarray, n0: int) -> np.ndarray:
    """
    JIT-compiled forward tracking lookup that reconstructs the optimal state trajectory.
    """
    T = len(P_d)
    n_control = np.zeros(T, dtype=np.int32)
    
    idx_p = np.abs(p_vals - P_d[0]).argmin()
    idx_n = np.abs(n_vals - n0).argmin()
    n_control[0] = n_vals[idx_n]
    
    for t in range(1, T):
        # MATLAB Bug 2: Delayed forward pass lookup sequence
        # Evaluates the optimal action using the previous demand state (t-1)
        idx_n = policy[t - 1, idx_p, idx_n]
        n_control[t] = n_vals[idx_n]
        
        # Updates demand state index for the NEXT loop iteration
        idx_p = np.abs(p_vals - P_d[t]).argmin()
        
    return n_control


class StochasticControl(ControlLaw):
    """
    Stochastic Dynamic Programming (SDP) tabular control policy framework.
    """
    def __init__(self, k_s: float, p_star: float, states: np.ndarray, 
                 transition_matrix: np.ndarray, n_vals: np.ndarray):
        self.k_s = k_s
        self.p_star = p_star
        self.states = states
        self.transition_matrix = transition_matrix
        self.n_vals = n_vals
        self.policy = None

    def compute(self, P_d: np.ndarray, n0: int) -> np.ndarray:
        T = len(P_d)
        self.policy = _solve_bellman_recursion(
            T, self.states, self.n_vals, self.transition_matrix, self.k_s, self.p_star
        )
        return _forward_pass_lookup(
            P_d, self.states, self.n_vals, self.policy, n0
        )