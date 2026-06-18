# python/src/controllers/threshold.py
import numpy as np
from numba import njit
from src.controllers.base import ControlLaw
from config import SimConfig

@njit(cache=True)
def _compute_threshold_core(P_d: np.ndarray, n0: int, k_s: float, p_nom: float, sigma: float) -> np.ndarray:
    """
    Numba implementation of the moving horizon threshold control strategy.
    Handles the index offset conversion from MATLAB's 1-based loop structure.
    """
    T = len(P_d)
    n_control = np.zeros(T, dtype=np.int32)
    n_control[0] = n0
    
    for t in range(1, T):
        # Calculate current power mismatch relative to the previous step's module capacity
        x = P_d[t] / p_nom - n_control[t-1]
        
        # Replicate MATLAB's remaining time index logic: max(1, T - t_matlab)
        # Since t is 0-indexed, t_matlab = t + 1, so T - t_matlab = T - t - 1
        t_rem = max(1.0, float(T - t - 1))
        x_thres = 2.0 * sigma * (n_control[t-1] * k_s / t_rem + 1.0)
        
        # Evaluate directional step boundaries
        if x > x_thres:
            n_control[t] = n_control[t-1] + 1
        elif x < -x_thres:
            n_control[t] = n_control[t-1] - 1
        else:
            n_control[t] = n_control[t-1]
            
    return n_control

class ThresholdControl(ControlLaw):
    """
    Heuristic moving-threshold tracking mechanism.
    Mirrors ThresholdControl inside ThresholdStochasticControl.m.
    """
    def __init__(self, config: SimConfig):
        self.config = config

    def compute(self, P_d: np.ndarray, n0: int) -> np.ndarray:
        return _compute_threshold_core(
            P_d, n0, self.config.k_s, self.config.p_nom, self.config.sigma
        )