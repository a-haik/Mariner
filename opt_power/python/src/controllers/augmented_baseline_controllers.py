# python/src/controllers/augmented_hybrid_controllers.py
import numpy as np
from src.controllers.base import ControlLaw
from src.core import State, Action
from src.utils.math_utils import nearest_index_1d

class AugmentedSDPBaselineControl(ControlLaw):
    """
    FC-Only Baseline Control Law, upgraded for the 3D Augmented state space.
    (Demand, Previous Modules, Previous FC Power).
    """
    def __init__(self, p_grid: np.ndarray, n_vals: np.ndarray, pfc_vals: np.ndarray, policy_matrix: np.ndarray):
        self.p_grid = p_grid
        self.n_vals = n_vals
        self.pfc_vals = pfc_vals
        self.policy = policy_matrix

    def get_action(self, state: State, time_sec: float) -> Action:
        macro_idx = int(time_sec // self.config.delta_t)
        t_idx = min(macro_idx, len(self.policy) - 1)

        # 1. Snap 3D continuous sensors to discrete grid
        idx_p = nearest_index_1d(self.p_grid, state.P_d)
        idx_n = nearest_index_1d(self.n_vals, float(state.n_prev))
        idx_pfc = nearest_index_1d(self.pfc_vals, state.p_fc_prev)
        
        # 2. Extract discrete module intent
        best_n_idx = self.policy[t_idx, idx_p, idx_n, idx_pfc]
        n_opt = int(self.n_vals[best_n_idx])
        
        return Action(n_modules=n_opt, p_batt=0.0, p_fc=state.P_d)