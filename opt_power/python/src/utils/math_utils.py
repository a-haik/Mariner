# python/src/utils/math_utils.py
import numpy as np
from numba import njit

@njit(cache=True)
def nearest_index_1d(grid: np.ndarray, val: float) -> int:
    """
    Finds the index of the closest value in a sorted 1D grid.
    Uses binary search (searchsorted) for O(log N) performance.
    """
    if val <= grid[0]:
        return 0
    if val >= grid[-1]:
        return len(grid) - 1
        
    idx = np.searchsorted(grid, val)
    
    # Check which boundary is closer
    if abs(grid[idx] - val) < abs(grid[idx - 1] - val):
        return idx
    else:
        return idx - 1

@njit(cache=True)
def linear_interp_1d(grid: np.ndarray, values: np.ndarray, val: float) -> float:
    """
    Performs 1D linear interpolation for continuous state mapping.
    Safely clamps to boundary values if the query is out of bounds.
    """
    if val <= grid[0]:
        return values[0]
    if val >= grid[-1]:
        return values[-1]
        
    idx = np.searchsorted(grid, val)
    x0, x1 = grid[idx - 1], grid[idx]
    y0, y1 = values[idx - 1], values[idx]
    
    # Calculate linear weight
    return y0 + (val - x0) * (y1 - y0) / (x1 - x0)

@njit(cache=True)
def bilinear_interp(x_grid: np.ndarray, y_grid: np.ndarray, values_2d: np.ndarray, 
                    x_val: float, y_val: float) -> float:
    """
    Performs 2D bilinear interpolation over a grid.
    Designed for the Hybrid Model to interpolate across continuous 
    Demand (P_d) and continuous State of Charge (SoC).
    """
    # X-axis clamping and fractional weight mapping
    if x_val <= x_grid[0]:
        x_idx0 = x_idx1 = 0
        x_w = 0.0
    elif x_val >= x_grid[-1]:
        x_idx0 = x_idx1 = len(x_grid) - 1
        x_w = 0.0
    else:
        x_idx1 = np.searchsorted(x_grid, x_val)
        x_idx0 = x_idx1 - 1
        x_w = (x_val - x_grid[x_idx0]) / (x_grid[x_idx1] - x_grid[x_idx0])

    # Y-axis clamping and fractional weight mapping
    if y_val <= y_grid[0]:
        y_idx0 = y_idx1 = 0
        y_w = 0.0
    elif y_val >= y_grid[-1]:
        y_idx0 = y_idx1 = len(y_grid) - 1
        y_w = 0.0
    else:
        y_idx1 = np.searchsorted(y_grid, y_val)
        y_idx0 = y_idx1 - 1
        y_w = (y_val - y_grid[y_idx0]) / (y_grid[y_idx1] - y_grid[y_idx0])

    # Retrieve the 4 corner values
    c00 = values_2d[x_idx0, y_idx0]
    c10 = values_2d[x_idx1, y_idx0]
    c01 = values_2d[x_idx0, y_idx1]
    c11 = values_2d[x_idx1, y_idx1]

    # Interpolate along the X-axis first
    c0 = c00 + x_w * (c10 - c00)
    c1 = c01 + x_w * (c11 - c01)

    # Interpolate the resulting points along the Y-axis
    return c0 + y_w * (c1 - c0)