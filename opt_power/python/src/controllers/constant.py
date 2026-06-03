# python/src/controllers/constant.py
import numpy as np
from numba import njit
from src.controllers.base import ControlLaw

@njit(cache=True)
def _compute_constant_core(horizon_length: int, n0: int) -> np.ndarray:
    """Fast compiled row generator to match repmat behavior."""
    return np.full(horizon_length, n0, dtype=np.int32)

class ConstantControl(ControlLaw):
    """
    Keeps the number of active modules invariant across changes in load demand.
    Mirrors ConstantControl.m.
    """
    def compute(self, P_d: np.ndarray, n0: int) -> np.ndarray:
        return _compute_constant_core(len(P_d), n0)