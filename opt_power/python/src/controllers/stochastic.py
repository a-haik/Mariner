# python/src/controllers/stochastic.py
import numpy as np
from src.controllers.base import ControlLaw
from src.core import State, Action

class StochasticControl(ControlLaw):
    """
    Pure online execution agent that uses a precalculated 
    SDP policy lookup matrix to make decisions step-by-step.
    """
    def __init__(self, states: np.ndarray, n_vals: np.ndarray, policy_matrix: np.ndarray):
        self.states = states
        self.n_vals = n_vals
        self.policy = policy_matrix
        self.current_step = 0

    def get_action(self, state: State) -> Action:
        # Map the continuous physical state to the closest discrete grid indices
        idx_p = np.abs(self.states - state.P_d).argmin()
        idx_n = np.abs(self.n_vals - state.n_prev).argmin()
        
        # Policy dimension is (T, p_size, n_size). Prevent out-of-bounds errors.
        t_idx = min(self.current_step, len(self.policy) - 1)
        
        best_n_idx = self.policy[t_idx, idx_p, idx_n]
        n_action = int(self.n_vals[best_n_idx])
        
        self.current_step += 1
        
        # Returns the generic Action object. (Battery split remains 0 for baseline)
        return Action(n_modules=n_action, p_batt=0.0)