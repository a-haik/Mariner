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
    get_c_min_kwh
)

@njit(cache=True)
def _solve_augmented_bellman(T: int, p_vals: np.ndarray, n_vals: np.ndarray, soc_vals: np.ndarray,
                             pfc_vals: np.ndarray, pb_vals: np.ndarray, transition_matrix: np.ndarray, 
                             Ts: float, p_max: float, p_nom: float, k_fc: float, k_h2: float, S_max: float, 
                             tau_fc: float, alpha_fc: float, a0: float, a1: float, a2: float,
                             C_bat: float, C_rep: float, E_life: float, lambda_trans: float,
                             soc_min: float, soc_max: float, nT: int, apply_terminal_n_cost: bool,
                             soc_target: float, apply_terminal_soc_cost: bool):
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
    V = np.full((T + 1, p_size, n_size, soc_size, pfc_size), np.inf, dtype=np.float64)
    policy_n = np.zeros((T, p_size, n_size, soc_size, pfc_size), dtype=np.int32)
    policy_pbatt = np.zeros((T, p_size, n_size, soc_size, pfc_size), dtype=np.float64)
    
    # Precalculate optimal charging cost for the boundary condition
    c_min_kwh = get_c_min_kwh(p_max, p_nom, tau_fc, alpha_fc, k_fc, k_h2, a0, a1, a2)

    # 1. NEW Terminal Boundary Condition (t = T) - NO Transient Cost
    for i in range(p_size):
        for j in range(n_size):
            n_val = n_vals[j]
            term_n_cost = 0.0
            if apply_terminal_n_cost:
                term_n_cost = calc_cost_switching(nT, n_val, k_fc, S_max) # Delegated to Cost Engine
                
            for k in range(soc_size):
                soc_val = soc_vals[k]
                term_soc_cost = 0.0
                if apply_terminal_soc_cost:
                    delta_e_kwh = (soc_target - soc_val) * C_bat
                    term_soc_cost = delta_e_kwh * c_min_kwh
                    
                for l in range(pfc_size):
                    V[T, i, j, k, l] = term_n_cost + term_soc_cost

    # Pre-allocate cache for stochastic expected future costs: [p_d, n_next, soc_next, pfc_next]
    exp_future_cache = np.zeros((p_size, n_size, soc_size, pfc_size), dtype=np.float64)

    # 2. Unified Backward Iteration (T-1 down to 0)
    for t in range(T - 1, -1, -1):
        
        # --- EXPECTATION TRANSPOSITION ---
        for i in range(p_size):
            for a_idx in range(n_size):
                for k_next in range(soc_size):
                    for l_next in range(pfc_size):
                        s = 0.0
                        for i_next in range(p_size):
                            s += transition_matrix[i, i_next] * V[t + 1, i_next, a_idx, k_next, l_next]
                        exp_future_cache[i, a_idx, k_next, l_next] = s
                    
        # --- STATE SEARCH ---
        for i in range(p_size):
            p_val = p_vals[i]
            for j in range(n_size):
                n_prev = n_vals[j]
                for k in range(soc_size):
                    soc_prev = soc_vals[k]
                    for l in range(pfc_size):
                        pfc_prev = pfc_vals[l]
                        
                        best_cost = np.inf
                        best_n_idx = 0  
                        best_pbatt = 0.0
                        
                        # --- ACTION SEARCH ---
                        for a_idx in range(n_size):
                            n_curr = n_vals[a_idx]
                            
                            for pb_idx in range(pbatt_size):
                                pbatt = pb_vals[pb_idx]
                                p_fc_curr = p_val - pbatt
                                
                                # 1. Hardware Limits Filtering
                                if p_fc_curr < 0:
                                    continue # Fuel cells can't sink power
                                    
                                if n_curr > 0 and (p_fc_curr / n_curr) > p_max:
                                    continue # Module Overload
                                    
                                if n_curr == 0 and p_fc_curr > 0:
                                    continue # Can't draw power if all modules are off
                                
                                # 2. State Kinematics (Project SoC)
                                soc_curr = soc_prev - (pbatt * (Ts / 3600.0)) / C_bat
                                if soc_curr < soc_min or soc_curr > soc_max:
                                    continue # Battery bounds
                                    
                                # 3. Centralized Cost Engine Calculations
                                c_o = calc_cost_operational(n_curr, p_fc_curr, p_nom, tau_fc, alpha_fc, k_fc, k_h2, a0, a1, a2, Ts)
                                c_s = calc_cost_switching(n_curr, n_prev, k_fc, S_max)
                                c_bat = calc_cost_battery(pbatt, Ts, C_rep, E_life)
                                c_trans = calc_cost_transient(n_curr, n_prev, p_fc_curr, pfc_prev, lambda_trans)
                                
                                # 4. Interpolate Expected Future Cost (2D over SoC and P_fc grids)
                                z_mat = exp_future_cache[i, a_idx, :, :]
                                exp_future = bilinear_interp_2d(soc_vals, pfc_vals, z_mat, soc_curr, p_fc_curr)
                                
                                total_cost = c_o + c_s + c_bat + c_trans + exp_future
                                
                                if total_cost < best_cost:
                                    best_cost = total_cost
                                    best_n_idx = a_idx
                                    best_pbatt = pbatt

                        V[t, i, j, k, l] = best_cost
                        policy_n[t, i, j, k, l] = best_n_idx
                        policy_pbatt[t, i, j, k, l] = best_pbatt
                        
    return policy_n, policy_pbatt, V


class AugmentedHybridSDPSolver:
    """Orchestrates the offline generation of the 4D Bellman matrices."""
    def __init__(self, config: SimConfig, mc_model: dict):
        self.config = config
        self.mc_model = mc_model

    def compute_solution(self, horizon_length: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # We assume you added pfc_vals to your config!
        return _solve_augmented_bellman(
            T=horizon_length,
            p_vals=self.mc_model['levels'],
            n_vals=self.config.n_vals,
            soc_vals=self.config.soc_vals,
            pfc_vals=self.config.pfc_vals,  # <--- NEW
            pb_vals=self.config.pb_vals,
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
            C_bat=float(self.config.C_bat),
            C_rep=float(self.config.C_rep),
            E_life=float(self.config.E_life),
            lambda_trans=float(self.config.lambda_trans), # <--- NEW
            soc_min=float(self.config.soc_min),
            soc_max=float(self.config.soc_max),
            nT=int(self.config.nT),                                   
            apply_terminal_n_cost=bool(self.config.apply_terminal_n_cost),
            soc_target=float(self.config.soc_target),
            apply_terminal_soc_cost=bool(self.config.apply_terminal_soc_cost)
        )