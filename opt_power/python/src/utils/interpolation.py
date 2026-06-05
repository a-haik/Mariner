# python/src/utils/interpolation.py
import numpy as np
from numba import njit

@njit(cache=True)
def linear_interp_1d(x: float, grid: np.ndarray, values: np.ndarray) -> float:
    """
    Ultra-fast 1D linear interpolation for continuous state mapping (SoC).
    Specifically designed for inner-loop performance inside Numba DP routines.
    """
    # Handle out-of-bounds mapping dynamically
    if x <= grid[0]:
        return values[0]
    if x >= grid[-1]:
        return values[-1]
        
    # Locate bounding grid indices
    idx = np.searchsorted(grid, x) - 1
    x0, x1 = grid[idx], grid[idx + 1]
    v0, v1 = values[idx], values[idx + 1]
    
    # Calculate weighted expected cost
    weight = (x - x0) / (x1 - x0)
    return v0 + weight * (v1 - v0)