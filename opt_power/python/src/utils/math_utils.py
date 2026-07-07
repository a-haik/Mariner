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
def get_exact_index_1d(val: float, grid_min: float, step_size: float, max_idx: int) -> int:
    """
    O(1) exact integer index calculation for perfectly uniform smart grids.
    Bypasses searchsorted by directly computing the algebraic index.
    Safely clamps to the grid boundaries.
    """
    if step_size <= 0.0:
        return 0
        
    # Calculate index and round to handle floating point drift
    idx = int(np.round((val - grid_min) / step_size))
    
    # Clamp to boundaries
    if idx < 0:
        return 0
    if idx > max_idx:
        return max_idx
    return idx

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

@njit(cache=True)
def bilinear_interp_2d(x_vals: np.ndarray, y_vals: np.ndarray, z_matrix: np.ndarray, 
                       x_target: float, y_target: float) -> float:
    """
    High-speed 2D bilinear interpolation for the 4D Bellman solver.
    x_vals: SoC grid (1D)
    y_vals: P_fc grid (1D)
    z_matrix: Expected future costs (2D array of shape [len(x_vals), len(y_vals)])
    """
    # 1. Clamp targets to grid boundaries (prevent extrapolation errors)
    x_t = max(x_vals[0], min(x_vals[-1], x_target))
    y_t = max(y_vals[0], min(y_vals[-1], y_target))
    
    # 2. Find indices using searchsorted
    x_idx = np.searchsorted(x_vals, x_t, side='right') - 1
    y_idx = np.searchsorted(y_vals, y_t, side='right') - 1
    
    x_idx = max(0, min(len(x_vals) - 2, x_idx))
    y_idx = max(0, min(len(y_vals) - 2, y_idx))
    
    # 3. Extract boundary coordinates
    x1, x2 = x_vals[x_idx], x_vals[x_idx + 1]
    y1, y2 = y_vals[y_idx], y_vals[y_idx + 1]
    
    # 4. Extract matrix values
    q11 = z_matrix[x_idx, y_idx]
    q21 = z_matrix[x_idx + 1, y_idx]
    q12 = z_matrix[x_idx, y_idx + 1]
    q22 = z_matrix[x_idx + 1, y_idx + 1]
    
    # 5. Interpolate
    denom = (x2 - x1) * (y2 - y1)
    if denom == 0.0:
        return q11 # Safety fallback
        
    f_xy = (q11 * (x2 - x_t) * (y2 - y_t) +
            q21 * (x_t - x1) * (y2 - y_t) +
            q12 * (x2 - x_t) * (y_t - y1) +
            q22 * (x_t - x1) * (y_t - y1)) / denom
            
    return f_xy

@njit(cache=True)
def calc_cost_operational(n_active: int, p_fc: float, p_nom: float, tau_fc: float, 
                          alpha_fc: float, k_fc: float, k_h2: float, 
                          a0: float, a1: float, a2: float, dt: float) -> float:
    """Calculates continuous H2 consumption and baseline electrochemical degradation."""
    
    if p_fc < 1e-5:
        p_fc = 0.0
        
    if n_active <= 0:
        return np.inf if p_fc > 0 else 0.0
        
    p_module = p_fc / n_active
    m_dot_h2 = a0 + a1 * p_module + a2 * (p_module ** 2)
    d_fc = (1.0 / (3600.0 * tau_fc)) * (1.0 + alpha_fc * ((p_module - p_nom) ** 2) / (p_nom ** 2))
    c_o_rate = (k_h2 * m_dot_h2 / 1000.0) + (k_fc * d_fc)
    
    return n_active * c_o_rate * dt

@njit(cache=True)
def calc_cost_switching(n_active: int, n_prev: int, k_fc: float, S_max: float) -> float:
    """Calculates the discrete start/stop penalty for modules."""
    k_s = k_fc / S_max
    return k_s * abs(n_active - n_prev)

@njit(cache=True)
def calc_cost_battery(p_batt: float, dt: float, C_rep: float, E_life: float) -> float:
    """Calculates the bidirectional wear-and-tear on the battery pack."""
    return C_rep * (abs(p_batt) * (dt / 3600.0)) / E_life

@njit(cache=True)
def calc_cost_transient(n_curr: int, n_prev: int, p_fc_curr: float, p_fc_prev: float, lambda_trans: float) -> float:
    """
    Calculates the transient penalty based on power fluctuation per active module.
    Ignores startup/shutdown penalties as they are handled by calc_cost_switching.
    """
    if n_curr <= 0:
        return 0.0
        
    p_curr = p_fc_curr / n_curr
    
    # If the system was off previously, p_prev is 0. 
    # To avoid double-penalizing the initial startup from a dead state,
    # we enforce 0 transient cost if n_prev was 0.
    if n_prev <= 0:
        return 0.0
        
    p_prev = p_fc_prev / n_prev
    
    return lambda_trans * n_curr * abs(p_curr - p_prev)

@njit(cache=True)
def get_c_min_kwh(p_max: float, p_nom: float, tau_fc: float, alpha_fc: float, 
                   k_fc: float, k_h2: float, a0: float, a1: float, a2: float) -> float:
    """
    Finds the absolute cheapest cost to produce 1 kWh of energy using the fuel cell.
    Used for the terminal boundary penalty calculation.
    """
    min_cost_per_kwh = np.inf
    # Discretize the power range to find the optimal specific consumption point
    for p in np.linspace(1.0, p_max, 200):
        m_dot = a0 + a1 * p + a2 * (p**2)
        d_fc = (1.0 / (3600.0 * tau_fc)) * (1.0 + alpha_fc * ((p - p_nom)**2) / (p_nom**2))
        cost_rate_sec = (k_h2 * m_dot / 1000.0) + (k_fc * d_fc)
        
        # Convert €/s to €/kWh
        cost_kwh = cost_rate_sec * 3600.0 / p
        if cost_kwh < min_cost_per_kwh:
            min_cost_per_kwh = cost_kwh
            
    return min_cost_per_kwh

@njit(cache=True)
def calculate_continuous_bounds(P_d_real: float, soc_real: float, n_next: int, Ts: float, 
                                p_max: float, Q_bat: float, soc_min: float, soc_max: float, 
                                pb_min_config: float, pb_max_config: float):
    """Calculates strict physical bounds for continuous P_batt to prevent the optimizer from searching impossible states."""
    # 1. SoC bounds constraints
    pb_soc_min = (soc_real - soc_max) * 3600.0 * Q_bat / Ts
    pb_soc_max = (soc_real - soc_min) * 3600.0 * Q_bat / Ts

    # 2. FC bounds constraints (P_fc = P_d_real - P_batt)
    pb_fc_max = P_d_real # P_fc >= 0
    if n_next > 0:
        pb_fc_min = P_d_real - (n_next * p_max) # P_fc / n_next <= p_max
    else:
        pb_fc_min = P_d_real # If modules are off, P_fc MUST be 0

    valid_min = max(pb_min_config, pb_soc_min, pb_fc_min)
    valid_max = min(pb_max_config, pb_soc_max, pb_fc_max)

    return valid_min, valid_max

# ---------------------------------------------------------
# STANDARD GSS (For 3D Hybrid Control)
# ---------------------------------------------------------
@njit(cache=True)
def eval_cost_standard(pbatt: float, P_d_real: float, soc_real: float, n_next: int, n_prev: int, Ts: float, 
                       p_max: float, p_nom: float, k_fc: float, k_h2: float, S_max: float, tau_fc: float, 
                       alpha_fc: float, a0: float, a1: float, a2: float, Q_bat: float, C_rep: float, 
                       E_life: float, soc_vals: np.ndarray, exp_v_slice: np.ndarray):
    soc_next = soc_real - (pbatt * (Ts / 3600.0)) / Q_bat
    p_fc = P_d_real - pbatt
    
    if p_fc < 1e-5:
        p_fc = 0.0

    C_o = calc_cost_operational(n_next, p_fc, p_nom, tau_fc, alpha_fc, k_fc, k_h2, a0, a1, a2, Ts)
    C_s = calc_cost_switching(n_next, n_prev, k_fc, S_max)
    C_bat = calc_cost_battery(pbatt, Ts, C_rep, E_life)
    exp_future = linear_interp_1d(soc_vals, exp_v_slice, soc_next)

    return C_o + C_s + C_bat + exp_future

@njit(cache=True)
def gss_standard(pb_min: float, pb_max: float, tol: float, P_d_real: float, soc_real: float, n_next: int, n_prev: int, 
                 Ts: float, p_max: float, p_nom: float, k_fc: float, k_h2: float, S_max: float, tau_fc: float, 
                 alpha_fc: float, a0: float, a1: float, a2: float, Q_bat: float, C_rep: float, E_life: float, 
                 soc_vals: np.ndarray, exp_v_slice: np.ndarray):
    if pb_max - pb_min <= tol:
        pb_opt = (pb_min + pb_max) / 2.0
        return pb_opt, eval_cost_standard(pb_opt, P_d_real, soc_real, n_next, n_prev, Ts, p_max, p_nom, k_fc, k_h2, S_max, tau_fc, alpha_fc, a0, a1, a2, Q_bat, C_rep, E_life, soc_vals, exp_v_slice)

    INVPHI = (np.sqrt(5.0) - 1.0) / 2.0
    INVPHI2 = (3.0 - np.sqrt(5.0)) / 2.0

    a = pb_min
    b = pb_max
    h = b - a
    c = a + INVPHI2 * h
    d = a + INVPHI * h

    cost_c = eval_cost_standard(c, P_d_real, soc_real, n_next, n_prev, Ts, p_max, p_nom, k_fc, k_h2, S_max, tau_fc, alpha_fc, a0, a1, a2, Q_bat, C_rep, E_life, soc_vals, exp_v_slice)
    cost_d = eval_cost_standard(d, P_d_real, soc_real, n_next, n_prev, Ts, p_max, p_nom, k_fc, k_h2, S_max, tau_fc, alpha_fc, a0, a1, a2, Q_bat, C_rep, E_life, soc_vals, exp_v_slice)

    while abs(h) > tol:
        if cost_c < cost_d:
            b = d
            d = c
            cost_d = cost_c
            h = b - a
            c = a + INVPHI2 * h
            cost_c = eval_cost_standard(c, P_d_real, soc_real, n_next, n_prev, Ts, p_max, p_nom, k_fc, k_h2, S_max, tau_fc, alpha_fc, a0, a1, a2, Q_bat, C_rep, E_life, soc_vals, exp_v_slice)
        else:
            a = c
            c = d
            cost_c = cost_d
            h = b - a
            d = a + INVPHI * h
            cost_d = eval_cost_standard(d, P_d_real, soc_real, n_next, n_prev, Ts, p_max, p_nom, k_fc, k_h2, S_max, tau_fc, alpha_fc, a0, a1, a2, Q_bat, C_rep, E_life, soc_vals, exp_v_slice)

    pb_opt = (a + b) / 2.0
    return pb_opt, eval_cost_standard(pb_opt, P_d_real, soc_real, n_next, n_prev, Ts, p_max, p_nom, k_fc, k_h2, S_max, tau_fc, alpha_fc, a0, a1, a2, Q_bat, C_rep, E_life, soc_vals, exp_v_slice)

# ---------------------------------------------------------
# AUGMENTED GSS (For 4D Augmented Control)
# ---------------------------------------------------------
@njit(cache=True)
def eval_cost_augmented(pbatt: float, P_d_real: float, soc_real: float, n_next: int, n_prev: int, pfc_prev: float, Ts: float, 
                        p_max: float, p_nom: float, k_fc: float, k_h2: float, S_max: float, tau_fc: float, alpha_fc: float, 
                        a0: float, a1: float, a2: float, Q_bat: float, C_rep: float, E_life: float, lambda_trans: float, 
                        soc_vals: np.ndarray, pfc_vals: np.ndarray, exp_v_slice: np.ndarray):
    soc_next = soc_real - (pbatt * (Ts / 3600.0)) / Q_bat
    p_fc = P_d_real - pbatt
    
    if p_fc < 1e-5:
        p_fc = 0.0

    C_o = calc_cost_operational(n_next, p_fc, p_nom, tau_fc, alpha_fc, k_fc, k_h2, a0, a1, a2, Ts)
    C_s = calc_cost_switching(n_next, n_prev, k_fc, S_max)
    C_bat = calc_cost_battery(pbatt, Ts, C_rep, E_life)
    C_trans = calc_cost_transient(n_next, n_prev, p_fc, pfc_prev, lambda_trans)
    
    exp_future = bilinear_interp_2d(soc_vals, pfc_vals, exp_v_slice, soc_next, p_fc)

    return C_o + C_s + C_bat + C_trans + exp_future

@njit(cache=True)
def gss_augmented(pb_min: float, pb_max: float, tol: float, P_d_real: float, soc_real: float, n_next: int, n_prev: int, pfc_prev: float,
                  Ts: float, p_max: float, p_nom: float, k_fc: float, k_h2: float, S_max: float, tau_fc: float, alpha_fc: float, 
                  a0: float, a1: float, a2: float, Q_bat: float, C_rep: float, E_life: float, lambda_trans: float, 
                  soc_vals: np.ndarray, pfc_vals: np.ndarray, exp_v_slice: np.ndarray):
    
    if pb_max - pb_min <= tol:
        pb_opt = (pb_min + pb_max) / 2.0
        return pb_opt, eval_cost_augmented(pb_opt, P_d_real, soc_real, n_next, n_prev, pfc_prev, Ts, p_max, p_nom, k_fc, k_h2, S_max, tau_fc, alpha_fc, a0, a1, a2, Q_bat, C_rep, E_life, lambda_trans, soc_vals, pfc_vals, exp_v_slice)

    INVPHI = (np.sqrt(5.0) - 1.0) / 2.0
    INVPHI2 = (3.0 - np.sqrt(5.0)) / 2.0
    a = pb_min
    b = pb_max
    h = b - a

    c = a + INVPHI2 * h
    d = a + INVPHI * h

    cost_c = eval_cost_augmented(c, P_d_real, soc_real, n_next, n_prev, pfc_prev, Ts, p_max, p_nom, k_fc, k_h2, S_max, tau_fc, alpha_fc, a0, a1, a2, Q_bat, C_rep, E_life, lambda_trans, soc_vals, pfc_vals, exp_v_slice)
    cost_d = eval_cost_augmented(d, P_d_real, soc_real, n_next, n_prev, pfc_prev, Ts, p_max, p_nom, k_fc, k_h2, S_max, tau_fc, alpha_fc, a0, a1, a2, Q_bat, C_rep, E_life, lambda_trans, soc_vals, pfc_vals, exp_v_slice)

    while abs(h) > tol:
        if cost_c < cost_d:
            b = d
            d = c
            cost_d = cost_c
            h = b - a
            c = a + INVPHI2 * h
            cost_c = eval_cost_augmented(c, P_d_real, soc_real, n_next, n_prev, pfc_prev, Ts, p_max, p_nom, k_fc, k_h2, S_max, tau_fc, alpha_fc, a0, a1, a2, Q_bat, C_rep, E_life, lambda_trans, soc_vals, pfc_vals, exp_v_slice)
        else:
            a = c
            c = d
            cost_c = cost_d
            h = b - a
            d = a + INVPHI * h
            cost_d = eval_cost_augmented(d, P_d_real, soc_real, n_next, n_prev, pfc_prev, Ts, p_max, p_nom, k_fc, k_h2, S_max, tau_fc, alpha_fc, a0, a1, a2, Q_bat, C_rep, E_life, lambda_trans, soc_vals, pfc_vals, exp_v_slice)

    pb_opt = (a + b) / 2.0
    return pb_opt, eval_cost_augmented(pb_opt, P_d_real, soc_real, n_next, n_prev, pfc_prev, Ts, p_max, p_nom, k_fc, k_h2, S_max, tau_fc, alpha_fc, a0, a1, a2, Q_bat, C_rep, E_life, lambda_trans, soc_vals, pfc_vals, exp_v_slice)