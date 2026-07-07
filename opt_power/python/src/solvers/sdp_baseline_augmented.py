# python/src/solvers/sdp_augmented_baseline.py
import numpy as np
from numba import njit
from typing import Tuple
from src.config import SimConfig
from src.utils.math_utils import (
    calc_cost_operational, 
    calc_cost_switching, 
    calc_cost_transient,
    linear_interp_1d,
    get_exact_index_1d
)

@njit(cache=True)
def _solve_augmented_baseline_bellman(T: int, p_vals: np.ndarray, n_vals: np.ndarray, pfc_vals: np.ndarray,
                                      transition_matrix: np.ndarray, Ts: float, p_max: float, p_nom: float, 
                                      k_fc: float, k_h2: float, S_max: float, tau_fc: float, alpha_fc: float, 
                                      a0: float, a1: float, a2: float, lambda_trans: float, 
                                      nT: int, apply_terminal_n_cost: bool,
                                      use_smart_grid: bool, dP: float):
    """
    JIT-compiled backward induction routine for the 3D FC-Only System.
    State space: [Demand, Previous Modules, Previous FC Power]
    """
    p_size = len(p_vals)
    n_size = len(n_vals)
    pfc_size = len(pfc_vals)
    
    # 3D Value Matrix and Policy Matrix
    V = np.full((T + 1, p_size, n_size, pfc_size), np.inf, dtype=np.float64)
    policy_n = np.zeros((T, p_size, n_size, pfc_size), dtype=np.int32)
    
    # 1. Terminal Boundary Condition (t = T)
    for i_idx in range(p_size):
        for j_idx in range(n_size):
            n_val = n_vals[j_idx]
            term_cost = 0.0
            if apply_terminal_n_cost:
                term_cost = calc_cost_switching(nT, n_val, k_fc, S_max)
                
            for l_idx in range(pfc_size):
                V[T, i_idx, j_idx, l_idx] = term_cost

    # Pre-allocate cache for stochastic expected future costs: [p_d, n_next, pfc_next]
    exp_future_cache = np.zeros((p_size, n_size, pfc_size), dtype=np.float64)

    # 2. Unified Backward Iteration
    for t in range(T - 1, -1, -1):
        
        # --- EXPECTATION TRANSPOSITION ---
        for i_idx in range(p_size):
            for a_idx in range(n_size):
                for l_next in range(pfc_size):
                    s = 0.0
                    for i_next in range(p_size):
                        s += transition_matrix[i_idx, i_next] * V[t + 1, i_next, a_idx, l_next]
                    exp_future_cache[i_idx, a_idx, l_next] = s
                    
        # --- STATE SEARCH ---
        for i_idx in range(p_size):
            p_val = p_vals[i_idx]
            for j_idx in range(n_size):
                n_prev = n_vals[j_idx]
                for l_idx in range(pfc_size):
                    pfc_prev = pfc_vals[l_idx]
                    
                    best_cost = np.inf
                    best_n_idx = 0  
                    
                    # --- ACTION SEARCH ---
                    for a_idx in range(n_size):
                        n_curr = n_vals[a_idx]
                        
                        # In FC-only, the Fuel Cell takes 100% of the raw demand
                        p_fc_curr = p_val  
                        
                        # Hardware Limits Filtering
                        if n_curr > 0 and (p_fc_curr / n_curr) > p_max:
                            continue # Module Overload
                        if n_curr == 0 and p_fc_curr > 0:
                            continue # Can't draw power if all modules are off
                            
                        # Centralized Cost Engine Calculations
                        c_o = calc_cost_operational(n_curr, p_fc_curr, p_nom, tau_fc, alpha_fc, k_fc, k_h2, a0, a1, a2, Ts)
                        c_s = calc_cost_switching(n_curr, n_prev, k_fc, S_max)
                        c_trans = calc_cost_transient(n_curr, n_prev, p_fc_curr, pfc_prev, lambda_trans)
                        
                        # 1D Interpolation over P_fc expected grid
                        if use_smart_grid:
                            pfc_idx = get_exact_index_1d(p_fc_curr, 0.0, dP, pfc_size - 1)
                            exp_future = exp_future_cache[i_idx, a_idx, pfc_idx]
                        else:
                            expected_vals_array = exp_future_cache[i_idx, a_idx, :]
                            exp_future = linear_interp_1d(pfc_vals, expected_vals_array, p_fc_curr)
                        
                        total_cost = c_o + c_s + c_trans + exp_future
                        
                        if total_cost < best_cost:
                            best_cost = total_cost
                            best_n_idx = a_idx

                    V[t, i_idx, j_idx, l_idx] = best_cost
                    policy_n[t, i_idx, j_idx, l_idx] = best_n_idx
                        
    return policy_n, V


class AugmentedBaselineSDPSolver:
    """Orchestrates the offline generation of the 3D Augmented Baseline matrices."""
    def __init__(self, config: SimConfig, mc_model: dict):
        self.config = config
        self.mc_model = mc_model

    def compute_solution(self, horizon_length: int) -> Tuple[np.ndarray, np.ndarray]:
        return _solve_augmented_baseline_bellman(
            T=horizon_length,
            p_vals=self.mc_model['levels'],
            n_vals=self.config.n_vals,
            pfc_vals=self.config.pfc_vals,
            transition_matrix=self.mc_model['P'],
            Ts=float(self.config.Ts),
            p_max=float(self.config.p_max),
            p_nom=float(self.config.p_nom),
            k_fc=float(self.config.k_fc),
            k_h2=float(self.config.k_h2),
            S_max=float(self.config.S_max),
            tau_fc=float(self.config.tau_fc),
            alpha_fc=float(self.config.alpha_fc),
            a0=float(self.config.a0),
            a1=float(self.config.a1),
            a2=float(self.config.a2),
            lambda_trans=float(self.config.lambda_trans),
            nT=int(self.config.nT),                                   
            apply_terminal_n_cost=bool(self.config.apply_terminal_n_cost),
            use_smart_grid=bool(self.config.use_smart_grid),
            dP=float(self.config.dP)
        )