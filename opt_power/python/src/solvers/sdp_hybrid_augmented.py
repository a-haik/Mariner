# python/src/solvers/sdp_hybrid_augmented.py
import numpy as np
from numba import njit
from typing import Tuple
from src.config import SimConfig
from src.utils.math_utils import (
    calc_cost_operational, 
    calc_cost_switching, 
    calc_cost_battery, 
    calc_cost_transient,
    bilinear_interp_2d, 
    get_c_min_kwh,
    get_exact_index_1d
)

@njit(cache=True)
def _solve_augmented_bellman(T: int, p_vals: np.ndarray, n_vals: np.ndarray, soc_vals: np.ndarray,
                             pfc_vals: np.ndarray, pb_vals: np.ndarray, transition_matrix: np.ndarray, 
                             delta_t: float, p_max: float, p_nom: float, k_fc: float, k_h2: float, S_max: float, 
                             tau_fc: float, alpha_fc: float, a0: float, a1: float, a2: float,
                             Q_b: float, C_rep: float, Q_eol: float, lambda_trans: float,
                             soc_min: float, soc_max: float, nT: int, apply_terminal_n_cost: bool,
                             soc_target: float, apply_terminal_soc_cost: bool,
                             use_smart_grid: bool, delta_P: float, dSoC: float):
    """
    JIT-compiled backward induction routine for the 4D Hybrid System.
    State space: [Demand, Previous Modules, Previous SoC, Previous FC Power]
    """
    p_size = len(p_vals)
    n_size = len(n_vals)
    soc_size = len(soc_vals)
    pfc_size = len(pfc_vals)
    pbatt_size = len(pb_vals)
    
    # 4D Value Matrix and Policy Matrices
    V = np.full((T + 1, p_size, n_size, soc_size, pfc_size), 0, dtype=np.float64)
    policy_n = np.zeros((T, p_size, n_size, soc_size, pfc_size), dtype=np.int32)
    policy_pbatt = np.zeros((T, p_size, n_size, soc_size, pfc_size), dtype=np.float64)
    
    # Precalculate optimal charging cost for the boundary condition
    c_min_kwh = get_c_min_kwh(p_max, p_nom, tau_fc, alpha_fc, k_fc, k_h2, a0, a1, a2)

    # 1. NEW Terminal Boundary Condition (t = T) - NO Transient Cost
    for i_idx in range(p_size):
        for j_idx in range(n_size):
            n_val = n_vals[j_idx]
            term_n_cost = 0.0
            if apply_terminal_n_cost:
                term_n_cost = calc_cost_switching(nT, n_val, k_fc, S_max) # Delegated to Cost Engine
                
            for k_idx in range(soc_size):
                soc_val = soc_vals[k_idx]
                term_soc_cost = 0.0
                if apply_terminal_soc_cost:
                    delta_e_kwh = (soc_target - soc_val) * Q_b
                    term_soc_cost = delta_e_kwh * c_min_kwh
                    
                for l_idx in range(pfc_size):
                    V[T, i_idx, j_idx, k_idx, l_idx] = term_n_cost + term_soc_cost

    # Pre-allocate cache for stochastic expected future costs: [p_d, n_next, soc_next, pfc_next]
    exp_future_cache = np.zeros((p_size, n_size, soc_size, pfc_size), dtype=np.float64)

    # 2. Unified Backward Iteration (T-1 down to 0)
    for t in range(T - 1, -1, -1):
        
        # --- EXPECTATION TRANSPOSITION ---
        for i_idx in range(p_size):
            for a_idx in range(n_size):
                for k_next in range(soc_size):
                    for l_next in range(pfc_size):
                        s = 0.0
                        for i_next in range(p_size):
                            s += transition_matrix[i_idx, i_next] * V[t + 1, i_next, a_idx, k_next, l_next]
                        exp_future_cache[i_idx, a_idx, k_next, l_next] = s
                    
        # --- STATE SEARCH ---
        for i_idx in range(p_size):
            p_val = p_vals[i_idx]
            for j_idx in range(n_size):
                n_prev = n_vals[j_idx]
                for k_idx in range(soc_size):
                    soc_prev = soc_vals[k_idx]
                    for l_idx in range(pfc_size):
                        pfc_prev = pfc_vals[l_idx]
                        
                        best_cost = 1e6
                        best_n_idx = 0  
                        best_pbatt = 0.0
                        
                        # --- ACTION SEARCH ---
                        for a_idx in range(n_size):
                            n_curr = n_vals[a_idx]
                            
                            for pb_idx in range(pbatt_size):
                                pbatt = pb_vals[pb_idx]
                                p_fc_curr = p_val - pbatt
                                
                                penalty = 0.0
                                
                                # 1. Hardware Limits Filtering (Slack Math & Clamping)
                                if p_fc_curr < 0:
                                    penalty += abs(p_fc_curr) * 1e6
                                    p_fc_curr = 0.0 # Clamp
                                    
                                if n_curr > 0 and (p_fc_curr / n_curr) > p_max:
                                    penalty += ((p_fc_curr / n_curr) - p_max) * 1e6
                                    p_fc_curr = n_curr * p_max # Clamp
                                    
                                if n_curr == 0 and p_fc_curr > 0:
                                    penalty += p_fc_curr * 1e6
                                    p_fc_curr = 0.0 # Clamp
                                
                                # 2. State Kinematics (Project SoC with Slack Math)
                                soc_curr = soc_prev - (pbatt * (delta_t / 3600.0)) / Q_b
                                
                                if soc_curr < soc_min:
                                    penalty += (soc_min - soc_curr) * 1e7
                                    soc_curr = soc_min # Clamp
                                elif soc_curr > soc_max:
                                    penalty += (soc_curr - soc_max) * 1e7
                                    soc_curr = soc_max # Clamp
                                    
                                # 3. Centralized Cost Engine Calculations
                                c_o = calc_cost_operational(n_curr, p_fc_curr, p_nom, tau_fc, alpha_fc, k_fc, k_h2, a0, a1, a2, delta_t)
                                c_s = calc_cost_switching(n_curr, n_prev, k_fc, S_max)
                                c_bat = calc_cost_battery(pbatt, delta_t, C_rep, Q_eol)
                                c_trans = calc_cost_transient(n_curr, n_prev, p_fc_curr, pfc_prev, lambda_trans)
                                
                                # 4. Expected Future Cost 
                                if use_smart_grid:
                                    soc_idx = get_exact_index_1d(soc_curr, soc_min, dSoC, soc_size - 1)
                                    pfc_idx = get_exact_index_1d(p_fc_curr, 0.0, delta_P, pfc_size - 1)
                                    exp_future = exp_future_cache[i_idx, a_idx, soc_idx, pfc_idx]
                                else:
                                    z_mat = exp_future_cache[i_idx, a_idx, :, :]
                                    exp_future = bilinear_interp_2d(soc_vals, pfc_vals, z_mat, soc_curr, p_fc_curr)
                                
                                # Add the proportional slack penalty to the total cost
                                total_cost = c_o + c_s + c_bat + c_trans + exp_future + penalty
                                
                                if total_cost < best_cost:
                                    best_cost = total_cost
                                    best_n_idx = a_idx
                                    best_pbatt = pbatt

                        V[t, i_idx, j_idx, k_idx, l_idx] = best_cost
                        policy_n[t, i_idx, j_idx, k_idx, l_idx] = best_n_idx
                        policy_pbatt[t, i_idx, j_idx, k_idx, l_idx] = best_pbatt
                        
    return policy_n, policy_pbatt, V


class AugmentedHybridSDPSolver:
    """Orchestrates the offline generation of the 4D Bellman matrices."""
    def __init__(self, config: SimConfig, mc_model: dict):
        self.config = config
        self.mc_model = mc_model

    def compute_solution(self, horizon_length: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        dSoC = (self.config.delta_P * self.config.delta_t) / (self.config.Q_b * 3600.0) if self.config.use_smart_grid else 0.0

        return _solve_augmented_bellman(
            T=horizon_length,
            p_vals=self.mc_model['levels'],
            n_vals=self.config.n_vals,
            soc_vals=self.config.soc_vals,
            pfc_vals=self.config.pfc_vals,
            pb_vals=self.config.pb_vals,
            transition_matrix=self.mc_model['P'],
            delta_t=float(self.config.delta_t),
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
            Q_b=float(self.config.Q_b),
            C_rep=float(self.config.C_rep),
            Q_eol=float(self.config.Q_eol),
            lambda_trans=float(self.config.lambda_trans),
            soc_min=float(self.config.soc_min),
            soc_max=float(self.config.soc_max),
            nT=int(self.config.nT),                                   
            apply_terminal_n_cost=bool(self.config.apply_terminal_n_cost),
            soc_target=float(self.config.soc_target),
            apply_terminal_soc_cost=bool(self.config.apply_terminal_soc_cost),
            use_smart_grid=bool(self.config.use_smart_grid),
            delta_P=float(self.config.delta_P),
            dSoC=float(dSoC)
        )