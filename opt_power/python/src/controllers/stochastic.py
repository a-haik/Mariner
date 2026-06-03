# src/controllers/stochastic.py
import numpy as np
from numba import njit
from src.controllers.base import ControlLaw

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
        # Evaluates the optimal action using the previous demand state (t-1)
        idx_n = policy[t - 1, idx_p, idx_n]
        n_control[t] = n_vals[idx_n]
        
        # Updates demand state index for the NEXT loop iteration
        idx_p = np.abs(p_vals - P_d[t]).argmin()
        
    return n_control

class StochasticControl(ControlLaw):
    """
    Pure online execution agent that uses a precalculated 
    SDP policy lookup matrix to make decisions.
    """
    def __init__(self, states: np.ndarray, n_vals: np.ndarray, policy_matrix: np.ndarray):
        self.states = states
        self.n_vals = n_vals
        self.policy = policy_matrix  # Injected pre-calculated decision mapping

    def compute(self, P_d: np.ndarray, n0: int) -> np.ndarray:
        """Executes the forward lookup pass over the power profile."""
        return _forward_pass_lookup(
            P_d, self.states, self.n_vals, self.policy, n0
        )