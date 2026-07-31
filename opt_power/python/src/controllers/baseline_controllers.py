# python/src/controllers/baseline_controllers.py
import numpy as np
from src.controllers.base import ControlLaw
from src.core import State, Action
from src.config import SimConfig
from src.utils.math_utils import nearest_index_1d

class BaselineSDPControl(ControlLaw):
    """
    Legacy 2D policy matrix (FC-only logic). 
    Assumes no battery exists for planning purposes.
    """
    def __init__(self, p_grid: np.ndarray, n_vals: np.ndarray, policy_matrix: np.ndarray):
        self.p_grid = p_grid
        self.n_vals = n_vals
        self.policy = policy_matrix

    def get_action(self, state: State, time_sec: float) -> Action:
        macro_idx = int(time_sec // self.config.delta_t)
        t_idx = min(macro_idx, len(self.policy) - 1)

        idx_p = nearest_index_1d(self.p_grid, state.P_d)
        idx_n = nearest_index_1d(self.n_vals, float(state.n_prev))
        
        best_n_idx = self.policy[t_idx, idx_p, idx_n]
        n_action = int(self.n_vals[best_n_idx])
        
        # Legacy logic: FC tries to handle the entire intended discrete demand
        discrete_p_d = self.p_grid[idx_p]
        p_fc_locked = discrete_p_d
        
        return Action(n_modules=n_action, p_batt=0.0, p_fc=p_fc_locked)

class BaselineThresholdControl(ControlLaw):
    """
    Heuristic moving-threshold tracking mechanism.
    """
    def __init__(self, config: SimConfig, horizon_length: int, sigma: float = 0.5):
        self.config = config
        self.T = horizon_length
        self.sigma = sigma

    def get_action(self, state: State, time_sec: float) -> Action:

        x = state.P_d / self.config.p_nom - state.n_prev
        t_rem = max(1.0, float(self.T - self.current_step - 1))
        k_s = self.config.k_fc / self.config.S_max
        x_thres = 2.0 * self.sigma * (state.n_prev * k_s / t_rem + 1.0)
        
        if x > x_thres:
            n_action = state.n_prev + 1
        elif x < -x_thres:
            n_action = state.n_prev - 1
        else:
            n_action = state.n_prev
            
        n_action = int(max(self.config.n_vals[0], min(self.config.n_vals[-1], n_action)))
        
        # FC targets the exact macro-demand block it sees
        p_fc_locked = state.P_d 
        
        return Action(n_modules=n_action, p_batt=0.0, p_fc=p_fc_locked)

class BaselineConstantControl(ControlLaw):
    """
    Keeps the number of active modules invariant across changes in load demand.
    """
    def __init__(self, config: SimConfig):
        self.n_constant = config.n0

    def get_action(self, state: State, time_sec: float) -> Action:
        # FC targets the exact macro-demand block it sees
        return Action(n_modules=self.n_constant, p_batt=0.0, p_fc=state.P_d)