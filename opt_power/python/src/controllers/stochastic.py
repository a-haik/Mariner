# python/src/controllers/stochastic.py
import numpy as np
from src.controllers.base import ControlLaw
from src.core import State, Action
from src.utils.math_utils import nearest_index_1d

class StochasticControl(ControlLaw):
    """
    Online execution agent that uses a precalculated 
    SDP policy matrix to make decisions via nearest-neighbor grid snapping.
    """
    def __init__(self, p_grid: np.ndarray, n_vals: np.ndarray, policy_matrix: np.ndarray):
        self.p_grid = p_grid
        self.n_vals = n_vals
        self.policy = policy_matrix
        self.current_step = 0

    def get_action(self, state: State) -> Action:
        # Snap continuous reality to the discrete grid
        idx_p = nearest_index_1d(self.p_grid, state.P_d)
        idx_n = nearest_index_1d(self.n_vals, float(state.n_prev))
        
        t_idx = min(self.current_step, len(self.policy) - 1)
        
        best_n_idx = self.policy[t_idx, idx_p, idx_n]
        n_action = int(self.n_vals[best_n_idx])
        
        self.current_step += 1
        return Action(n_modules=n_action, p_batt=0.0)