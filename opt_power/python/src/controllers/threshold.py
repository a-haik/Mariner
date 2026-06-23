# python/src/controllers/threshold.py
from src.controllers.base import ControlLaw
from src.core import State, Action
from config import SimConfig

class ThresholdControl(ControlLaw):
    """
    Heuristic moving-threshold tracking mechanism.
    Refactored to evaluate one State per discrete macro-step.
    """
    def __init__(self, config: SimConfig, horizon_length: int, sigma: float = 0.5):
        self.config = config
        self.T = horizon_length
        self.sigma = sigma
        self.current_step = 0

    def get_action(self, state: State) -> Action:
        # Calculate power mismatch relative to the previous step's module capacity
        x = state.P_d / self.config.p_nom - state.n_prev
        
        # Remaining time index logic
        t_rem = max(1.0, float(self.T - self.current_step - 1))
        
        # Calculate actual physical switching cost ratio
        k_s = self.config.k_fc / self.config.S_max
        
        x_thres = 2.0 * self.sigma * (state.n_prev * k_s / t_rem + 1.0)
        
        if x > x_thres:
            n_action = state.n_prev + 1
        elif x < -x_thres:
            n_action = state.n_prev - 1
        else:
            n_action = state.n_prev
            
        # Hard clip to the boundaries of the valid action space
        n_action = int(max(self.config.n_vals[0], min(self.config.n_vals[-1], n_action)))
        
        self.current_step += 1
        return Action(n_modules=n_action, p_batt=0.0)