# python/src/solvers/hybrid_sdp.py
import numpy as np
from numba import njit
from typing import Tuple
from src.config import SimConfig
from src.utils.math_utils import linear_interp_1d, get_c_min_kwh

@njit(cache=True)
def _solve_hybrid_bellman(T: int, p_vals: np.ndarray, n_vals: np.ndarray, soc_vals: np.ndarray,
                          pb_vals: np.ndarray, transition_matrix: np.ndarray, Ts: float, 
                          p_max: float, p_nom: float, k_fc: float, k_h2: float, S_max: float, 
                          tau_fc: float, alpha_fc: float, a0: float, a1: float, a2: float,
                          C_bat: float, C_rep: float, E_life: float, 
                          soc_min: float, soc_max: float, 
                          nT: int, apply_terminal_n_cost: bool,
                          soc_target: float, apply_terminal_soc_cost: bool):
    """
    JIT-compiled backward induction routine for the 4D Hybrid System.
    Updated with true T+1 terminal boundary conditions.
    """
    p_size = len(p_vals)
    n_size = len(n_vals)
    soc_size = len(soc_vals)
    pbatt_size = len(pb_vals)
    
    # EXPAND V matrix to size T+1. Policies stay T.
    V = np.full((T + 1, p_size, n_size, soc_size), np.inf, dtype=np.float64)
    policy_n = np.zeros((T, p_size, n_size, soc_size), dtype=np.int32)
    policy_pbatt = np.zeros((T, p_size, n_size, soc_size), dtype=np.float64)
    
    k_s = k_fc / S_max
    
    # 1. Precalculate optimal charging cost for the boundary condition
    c_min_kwh = get_c_min_kwh(p_max, p_nom, tau_fc, alpha_fc, k_fc, k_h2, a0, a1, a2)

    # 2. NEW Terminal Boundary Condition (t = T)
    for i in range(p_size):
        for j in range(n_size):
            n_val = n_vals[j]
            
            term_n_cost = 0.0
            if apply_terminal_n_cost:
                term_n_cost = k_s * abs(n_val - nT)
                
            for k in range(soc_size):
                soc_val = soc_vals[k]
                
                term_soc_cost = 0.0
                if apply_terminal_soc_cost:
                    # Positive cost for deficit, Negative cost (reward) for surplus
                    delta_e_kwh = (soc_target - soc_val) * C_bat
                    term_soc_cost = delta_e_kwh * c_min_kwh
                    
                V[T, i, j, k] = term_n_cost + term_soc_cost

    # Pre-allocated cache for the stochastic expected future costs over Demand
    exp_future_cache = np.zeros((p_size, n_size, soc_size), dtype=np.float64)

    # 3. Unified Backward Iteration (T-1 down to 0)
    for t in range(T - 1, -1, -1):
        
        # --- EXPECTATION TRANSPOSITION ---
        # Calculate Expected Value over the stochastic Demand dimension ONCE per timestep
        for i in range(p_size):
            for a_idx in range(n_size):
                for k_next in range(soc_size):
                    s = 0.0
                    for i_next in range(p_size):
                        s += transition_matrix[i, i_next] * V[t + 1, i_next, a_idx, k_next]
                    exp_future_cache[i, a_idx, k_next] = s
                    
        # --- STATE SEARCH ---
        for i in range(p_size):
            p_val = p_vals[i]
            for j in range(n_size):
                n_val = n_vals[j]
                if n_val <= 0:
                    continue
                for k in range(soc_size):
                    soc_val = soc_vals[k]
                    
                    best_cost = np.inf
                    best_n_idx = 0  
                    best_pbatt = 0.0
                    
                    # --- ACTION SEARCH ---
                    for a_idx in range(n_size):
                        n_next = n_vals[a_idx]
                        if n_next <= 0:
                            continue
                            
                        for pb_idx in range(pbatt_size):
                            pbatt = pb_vals[pb_idx]
                            
                            # 1. State Kinematics (Project SoC)
                            soc_next = soc_val - (pbatt * (Ts / 3600.0)) / C_bat
                            
                            # 2. Hardware / Physical Limit Filtering
                            if soc_next < soc_min or soc_next > soc_max:
                                continue 
                                
                            p_fc = p_val - pbatt
                            if p_fc < 0:
                                continue # Fuel cell cannot absorb reverse current
                                
                            p_module = p_fc / n_next
                            if p_module > p_max:
                                continue # Hardware overload
                                
                            # 3. Instantaneous Cost Calculations
                            m_dot_h2 = a0 + a1 * p_module + a2 * (p_module ** 2)
                            d_fc = (1.0 / (3600.0 * tau_fc)) * (1.0 + alpha_fc * ((p_module - p_nom) ** 2) / (p_nom ** 2))
                            c_o_rate = (k_h2 * m_dot_h2 / 1000.0) + (k_fc * d_fc)
                            
                            C_o = n_next * c_o_rate * Ts
                            C_s = k_s * abs(n_next - n_val)
                            C_bat_cost = C_rep * (abs(pbatt) * (Ts / 3600.0)) / E_life
                            
                            # 4. Interpolate Expected Future Cost (1D over SoC grid)
                            expected_vals_array = exp_future_cache[i, a_idx, :]
                            exp_future = linear_interp_1d(soc_vals, expected_vals_array, soc_next)
                            
                            total_cost = C_o + C_s + C_bat_cost + exp_future
                            
                            if total_cost < best_cost:
                                best_cost = total_cost
                                best_n_idx = a_idx
                                best_pbatt = pbatt

                    V[t, i, j, k] = best_cost
                    policy_n[t, i, j, k] = best_n_idx
                    policy_pbatt[t, i, j, k] = best_pbatt
                    
    return policy_n, policy_pbatt, V

class HybridSDPSolver:
    """Orchestrates the offline generation of the 4D Bellman matrices."""
    def __init__(self, config: SimConfig, mc_model: dict):
        self.config = config
        self.mc_model = mc_model

    def compute_solution(self, horizon_length: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
            policy_n: (T, P_size, n_size, soc_size) array of optimal module counts.
            policy_pbatt: (T, P_size, n_size, soc_size) array of optimal battery power [kW].
            V: (T, P_size, n_size, soc_size) array of expected cumulative costs.
        """
        return _solve_hybrid_bellman(
            T=horizon_length,
            p_vals=self.mc_model['levels'],
            n_vals=self.config.n_vals,
            soc_vals=self.config.soc_vals,
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
            soc_min=float(self.config.soc_min),
            soc_max=float(self.config.soc_max),
            nT=int(self.config.nT),                                   
            apply_terminal_n_cost=bool(self.config.apply_terminal_n_cost),
            soc_target=float(self.config.soc_target),
            apply_terminal_soc_cost=bool(self.config.apply_terminal_soc_cost)
        )